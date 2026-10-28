"""
Customer segmentation and lifetime value for a retention program.

Question: an online retailer wants to spend its retention budget where it matters.
Who are the best customers, who is slipping away, and what is each group worth?

Steps
1. Simulate two years of order data for 8,000 customers, including returns,
   test orders, missing customer IDs and duplicate rows.
2. Clean it and document every rule.
3. Build monthly acquisition cohorts and a retention curve.
4. Score customers on Recency, Frequency and Monetary value (RFM) and group
   them into named segments a marketing team can act on.
5. Estimate 12-month customer lifetime value from each customer's own order
   rate, spend and churn risk, and check it against what customers actually spent next.
6. Size each segment and recommend an action for it.

Run: python segmentation.py   (writes charts/ and results.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import _style as S  # noqa: E402

S.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

rng = np.random.default_rng(11)
N = 8_000
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
CUTOFF = pd.Timestamp("2024-12-31")  # model on year one, check on year two

# ---------- 1. Simulate ----------
types = rng.choice(["loyal", "regular", "occasional", "one_and_done"], N, p=[0.10, 0.25, 0.35, 0.30])
rate = {"loyal": 1.6, "regular": 0.7, "occasional": 0.25, "one_and_done": 0.04}  # orders per month
aov = {"loyal": 95, "regular": 70, "occasional": 60, "one_and_done": 55}
life = {"loyal": 30, "regular": 14, "occasional": 9, "one_and_done": 2}  # mean active months
first = START + pd.to_timedelta(rng.integers(0, 600, N), unit="D")
rows = []
for i in range(N):
    t = types[i]
    end = min(END, first[i] + pd.Timedelta(days=float(rng.exponential(life[t]) * 30.4)))
    d = first[i]
    while d <= end:
        rows.append((f"C{i:05d}", d, round(rng.gamma(4, aov[t] / 4), 2)))
        d += pd.Timedelta(days=float(rng.exponential(30.4 / rate[t])))
orders = pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])
orders["order_id"] = [f"O{i:07d}" for i in range(len(orders))]

# real-world mess
n = len(orders)
mess = orders.copy()
ret = mess.sample(frac=0.04, random_state=2).copy(); ret["amount"] *= -1; ret["order_id"] += "R"
test = mess.sample(60, random_state=3).copy(); test["amount"] = 0.01; test["order_id"] += "T"
mess = pd.concat([mess, ret, test, mess.sample(500, random_state=4)])
mess.loc[mess.sample(300, random_state=5).index, "customer_id"] = np.nan
raw = len(mess)

# ---------- 2. Clean ----------
log = {}
c = mess.drop_duplicates("order_id"); log["duplicate_orders_removed"] = raw - len(c)
before = len(c); c = c[c["customer_id"].notna()]; log["rows_missing_customer_id_removed"] = before - len(c)
before = len(c); c = c[c["amount"] != 0.01]; log["test_orders_removed"] = before - len(c)
returns = c[c["amount"] < 0]; log["returns_netted_against_orders"] = len(returns)
c = c[c["amount"] > 0].copy()
ret_map = returns.assign(order_id=returns["order_id"].str.rstrip("R")).set_index("order_id")["amount"]
ret_map = ret_map[~ret_map.index.duplicated()]
c["net_amount"] = c["amount"] + c["order_id"].map(ret_map).fillna(0)
c = c[c["net_amount"] > 0]
log["clean_orders"] = len(c)
c.to_csv(HERE / "data" / "orders_clean.csv.gz", index=False, compression="gzip")

# ---------- 3. Cohorts ----------
c["cohort"] = c.groupby("customer_id")["order_date"].transform("min").dt.to_period("M")
c["age"] = (c["order_date"].dt.to_period("M") - c["cohort"]).apply(lambda x: x.n)
coh = c.groupby(["cohort", "age"])["customer_id"].nunique().unstack(fill_value=0)
ret_curve = (coh.div(coh[0], axis=0)).iloc[:12, :13]

# ---------- 4. RFM on year one ----------
y1 = c[c["order_date"] <= CUTOFF]
y2 = c[c["order_date"] > CUTOFF]
r = y1.groupby("customer_id").agg(last=("order_date", "max"), first=("order_date", "min"),
                                   frequency=("order_id", "count"), monetary=("net_amount", "sum"))
r["recency"] = (CUTOFF - r["last"]).dt.days
r["tenure"] = (CUTOFF - r["first"]).dt.days + 1
r["R"] = pd.qcut(r["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
r["F"] = pd.qcut(r["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
r["M"] = pd.qcut(r["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)


def segment(row):
    R, F = row.R, row.F
    if R >= 4 and F >= 4: return "Champions"
    if R >= 3 and F >= 3: return "Loyal"
    if R >= 4 and F <= 2: return "New and promising"
    if R == 3 and F <= 2: return "Needs attention"
    if R <= 2 and F >= 3: return "At risk"
    return "Hibernating"


r["segment"] = r.apply(segment, axis=1)

# ---------- 5. 12-month CLV ----------
# expected orders = personal order rate x chance still active (decays with time since last order).
# The rate is shrunk toward the average customer so one early order does not look like a habit.
tenure_m = r["tenure"] / 30.4
pop_rate = r["frequency"].sum() / tenure_m.sum()
PRIOR_MONTHS = 6
rate_m = (r["frequency"] + pop_rate * PRIOR_MONTHS) / (tenure_m + PRIOR_MONTHS)
gap = (r["tenure"] / r["frequency"]).clip(lower=14)
p_active = np.exp(-r["recency"] / (1.5 * gap))
r["aov"] = r["monetary"] / r["frequency"]
r["clv_12m"] = rate_m * 12 * p_active * r["aov"]
actual = y2.groupby("customer_id")["net_amount"].sum()
r["actual_next_12m"] = r.index.map(actual).fillna(0)

seg = r.groupby("segment").agg(customers=("R", "size"), avg_clv=("clv_12m", "mean"),
                               total_clv=("clv_12m", "sum"), actual=("actual_next_12m", "sum"),
                               avg_orders=("frequency", "mean"), avg_recency=("recency", "mean"))
seg["share_customers"] = seg["customers"] / seg["customers"].sum()
seg["share_value"] = seg["total_clv"] / seg["total_clv"].sum()
seg = seg.sort_values("avg_clv", ascending=False)
ACTION = {
    "Champions": "Early access and referral asks. No discounts needed.",
    "Loyal": "Cross-sell and a loyalty tier to move them up.",
    "At risk": "Win-back series now, with a personal offer.",
    "New and promising": "Second-order nudge within 30 days of first purchase.",
    "Needs attention": "Reminder content and bestsellers, light incentive.",
    "Hibernating": "Low-cost reactivation only. Suppress from paid media.",
}

top20 = r.sort_values("clv_12m", ascending=False).head(int(len(r) * 0.2))
lift_top = top20["actual_next_12m"].mean() / r["actual_next_12m"].mean()
corr = float(np.corrcoef(r["clv_12m"], r["actual_next_12m"])[0, 1])
dec = pd.qcut(r["clv_12m"].rank(method="first"), 10, labels=range(1, 11))
dec_tbl = r.groupby(dec, observed=True).agg(pred=("clv_12m", "mean"), actual=("actual_next_12m", "mean"))

results = {
    "cleaning": log,
    "customers_scored": int(len(r)),
    "segments": {s: {"customers": int(v.customers), "share_customers": round(v.share_customers, 3),
                     "share_of_predicted_value": round(v.share_value, 3), "avg_clv_12m": round(v.avg_clv, 2),
                     "avg_orders_year1": round(v.avg_orders, 1), "avg_days_since_order": round(v.avg_recency),
                     "action": ACTION[s]} for s, v in seg.iterrows()},
    "validation": {"clv_vs_actual_correlation": round(corr, 2),
                   "top20pct_by_clv_spend_vs_average": round(lift_top, 2),
                   "predicted_total": round(r["clv_12m"].sum()), "actual_total": round(r["actual_next_12m"].sum())},
    "retention": {"month1": round(float(ret_curve[1].mean()), 3), "month3": round(float(ret_curve[3].mean()), 3),
                  "month6": round(float(ret_curve[6].mean()), 3), "month12": round(float(ret_curve[12].mean()), 3)},
}
(HERE / "results.json").write_text(json.dumps(results, indent=2))

# ---------- Charts ----------
out = HERE / "charts"
fig, ax = plt.subplots(figsize=(9, 4.2))
im = ax.imshow(ret_curve.values[:, 1:], cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
ax.set_xticks(range(12), [f"M{i}" for i in range(1, 13)])
ax.set_yticks(range(len(ret_curve)), [str(p) for p in ret_curve.index])
for (i, j), v in np.ndenumerate(ret_curve.values[:, 1:]):
    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > 0.3 else S.INK)
ax.grid(False)
ax.set_title("Share of each monthly cohort still buying, by months since first order")
fig.savefig(out / "cohort_retention.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.2))
s2 = seg.sort_values("share_value")
yy = np.arange(len(s2))
ax.barh(yy - 0.2, s2["share_customers"] * 100, 0.38, color="#B9C2DA", label="% of customers")
ax.barh(yy + 0.2, s2["share_value"] * 100, 0.38, color=S.COBALT, label="% of predicted 12-month value")
ax.set_yticks(yy, s2.index)
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_title("A small group holds most of the future value")
ax.legend(loc="lower right")
fig.savefig(out / "segments_value.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 11), dec_tbl["pred"], marker="o", color=S.COBALT, label="Predicted 12-month value")
ax.plot(range(1, 11), dec_tbl["actual"], marker="o", color=S.PEACH, label="Actual spend in year two")
ax.set_xticks(range(1, 11))
ax.set_xlabel("Customer decile by predicted value (10 = highest)")
ax.yaxis.set_major_formatter(FuncFormatter(S.money))
ax.set_title("The CLV score ranks customers the way they actually spent")
ax.legend()
fig.savefig(out / "clv_validation.png"); plt.close(fig)

print(json.dumps(results, indent=2))
