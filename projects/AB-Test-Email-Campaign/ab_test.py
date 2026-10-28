"""
A/B test readout for an email campaign, from planning to decision.

Question: a brand wants to test a new email (personalized subject line plus
free-shipping offer) against the current one. Should it roll out?

Steps
1. Plan: how many recipients are needed to detect a 10% lift in conversion?
2. Simulate the test send with a 50/50 split.
3. Check the data before trusting it: sample ratio mismatch, duplicate sends,
   and bot clicks from email security scanners.
4. Read the result three ways: a frequentist test, a Bayesian probability,
   and a bootstrap on revenue per recipient.
5. Look at segments carefully (with a correction for testing several at once).
6. Turn it into a decision with the expected annual revenue impact.

Run: python ab_test.py   (writes charts/ and results.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import _style as S  # noqa: E402

S.apply()
import matplotlib.pyplot as plt  # noqa: E402

rng = np.random.default_rng(2026)

# ---------- 1. Plan ----------
BASE_CR, MDE, ALPHA, POWER = 0.032, 0.10, 0.05, 0.80
p1, p2 = BASE_CR, BASE_CR * (1 + MDE)
z_a, z_b = stats.norm.ppf(1 - ALPHA / 2), stats.norm.ppf(POWER)
pbar = (p1 + p2) / 2
n_per_arm = int(np.ceil(((z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2))

# ---------- 2. Simulate the send ----------
N = 2 * n_per_arm + 4_000
seg = rng.choice(["New", "Returning", "VIP"], N, p=[0.45, 0.45, 0.10])
arm = rng.choice(["A", "B"], N)
base = {"New": 0.022, "Returning": 0.036, "VIP": 0.070}
lift = {"New": 0.20, "Returning": 0.08, "VIP": 0.00}  # true effects differ by segment
cr = np.array([base[s] * (1 + (lift[s] if a == "B" else 0)) for s, a in zip(seg, arm)])
conv = rng.random(N) < cr
order = np.where(conv, rng.gamma(3, 28, N), 0)
order = np.where(conv & (arm == "B"), order - 6.0, order)  # free shipping costs about $6 an order
df = pd.DataFrame({"user_id": np.arange(N), "arm": arm, "segment": seg, "converted": conv, "revenue": order.clip(0)})

# messy reality: re-sends and security-scanner clicks
df["is_bot"] = False
dups = df.sample(700, random_state=1)
bots = df.sample(1_100, random_state=2).assign(user_id=lambda d: d.user_id + 10_000_000, converted=False, revenue=0.0, is_bot=True)
raw = pd.concat([df, dups, bots], ignore_index=True)

# ---------- 3. Data checks ----------
checks = {"raw_rows": len(raw)}
clean = raw[~raw["is_bot"]]; checks["scanner_rows_removed"] = len(raw) - len(clean)
before = len(clean); clean = clean.drop_duplicates("user_id"); checks["duplicate_sends_removed"] = before - len(clean)
counts = clean["arm"].value_counts()
srm_p = stats.chisquare([counts["A"], counts["B"]]).pvalue
checks.update({"recipients_A": int(counts["A"]), "recipients_B": int(counts["B"]),
               "sample_ratio_p_value": round(float(srm_p), 3), "sample_ratio_ok": bool(srm_p > 0.01),
               "needed_per_arm": n_per_arm})

# ---------- 4. Read the result ----------
g = clean.groupby("arm").agg(n=("user_id", "size"), conv=("converted", "sum"), rev=("revenue", "sum"))
g["cr"] = g["conv"] / g["n"]
g["rpr"] = g["rev"] / g["n"]
pA, pB = g.loc["A", "cr"], g.loc["B", "cr"]
nA, nB = g.loc["A", "n"], g.loc["B", "n"]
se = np.sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
diff = pB - pA
ci = (diff - 1.96 * se, diff + 1.96 * se)
pool = (g["conv"].sum()) / (g["n"].sum())
z = diff / np.sqrt(pool * (1 - pool) * (1 / nA + 1 / nB))
p_val = 2 * (1 - stats.norm.cdf(abs(z)))

# Bayesian: Beta(1,1) prior on each conversion rate
postA = rng.beta(1 + g.loc["A", "conv"], 1 + nA - g.loc["A", "conv"], 200_000)
postB = rng.beta(1 + g.loc["B", "conv"], 1 + nB - g.loc["B", "conv"], 200_000)
prob_b_better = float((postB > postA).mean())
exp_loss_b = float(np.maximum(postA - postB, 0).mean())

# bootstrap revenue per recipient
ra = clean.loc[clean.arm == "A", "revenue"].values
rb = clean.loc[clean.arm == "B", "revenue"].values
boot = np.array([rng.choice(rb, len(rb)).mean() - rng.choice(ra, len(ra)).mean() for _ in range(2_000)])
rpr_ci = np.percentile(boot, [2.5, 97.5])

# ---------- 5. Segments with Holm correction ----------
segs = []
for s in ["New", "Returning", "VIP"]:
    d = clean[clean.segment == s].groupby("arm").agg(n=("user_id", "size"), c=("converted", "sum"))
    a_, b_ = d.loc["A"], d.loc["B"]
    pa, pb = a_.c / a_.n, b_.c / b_.n
    pp = (a_.c + b_.c) / (a_.n + b_.n)
    zz = (pb - pa) / np.sqrt(pp * (1 - pp) * (1 / a_.n + 1 / b_.n))
    sse = np.sqrt(pa * (1 - pa) / a_.n + pb * (1 - pb) / b_.n)
    segs.append({"segment": s, "cr_A": pa, "cr_B": pb, "rel_lift": pb / pa - 1,
                 "p_value": 2 * (1 - stats.norm.cdf(abs(zz))), "lo": (pb - pa - 1.96 * sse) / pa, "hi": (pb - pa + 1.96 * sse) / pa})
segs = pd.DataFrame(segs).sort_values("p_value").reset_index(drop=True)
m = len(segs)
segs["holm_threshold"] = [ALPHA / (m - i) for i in range(m)]
sig = True
flags = []
for _, row in segs.iterrows():
    sig = sig and row.p_value < row.holm_threshold
    flags.append(sig)
segs["significant_after_correction"] = flags

# revenue per recipient, new customers only
na = clean.loc[(clean.arm == "A") & (clean.segment == "New"), "revenue"].values
nb = clean.loc[(clean.arm == "B") & (clean.segment == "New"), "revenue"].values
boot_new = np.array([rng.choice(nb, len(nb)).mean() - rng.choice(na, len(na)).mean() for _ in range(2_000)])
new_ci = np.percentile(boot_new, [2.5, 97.5])
new_share = (clean.segment == "New").mean()

# ---------- 6. Decision ----------
ANNUAL_SENDS = 24 * 400_000  # 24 campaigns a year to a 400K list
annual_impact = (g.loc["B", "rpr"] - g.loc["A", "rpr"]) * ANNUAL_SENDS
annual_impact_lo, annual_impact_hi = rpr_ci[0] * ANNUAL_SENDS, rpr_ci[1] * ANNUAL_SENDS

results = {
    "plan": {"baseline_conversion": BASE_CR, "min_detectable_lift": MDE, "alpha": ALPHA, "power": POWER, "recipients_needed_per_arm": n_per_arm},
    "data_checks": checks,
    "overall": {"cr_A": round(pA, 4), "cr_B": round(pB, 4), "relative_lift": round(pB / pA - 1, 3),
                "abs_diff_ci95": [round(ci[0], 4), round(ci[1], 4)], "p_value": round(float(p_val), 4),
                "prob_B_better": round(prob_b_better, 3), "expected_loss_if_ship_B": round(exp_loss_b, 5),
                "revenue_per_recipient_A": round(g.loc["A", "rpr"], 3), "revenue_per_recipient_B": round(g.loc["B", "rpr"], 3),
                "rpr_diff_ci95": [round(rpr_ci[0], 3), round(rpr_ci[1], 3)]},
    "segments": segs.round(4).to_dict(orient="records"),
    "decision": {"annual_sends_assumed": ANNUAL_SENDS,
                 "ship_to_everyone": {"expected_annual_revenue_impact": round(annual_impact),
                                      "impact_range_95": [round(annual_impact_lo), round(annual_impact_hi)]},
                 "ship_to_new_customers_only": {
                     "rpr_diff": round(float(nb.mean() - na.mean()), 3),
                     "rpr_diff_ci95": [round(new_ci[0], 3), round(new_ci[1], 3)],
                     "expected_annual_revenue_impact": round((nb.mean() - na.mean()) * ANNUAL_SENDS * new_share),
                     "impact_range_95": [round(new_ci[0] * ANNUAL_SENDS * new_share), round(new_ci[1] * ANNUAL_SENDS * new_share)]},
                 "recommendation": "Roll out version B to new customers only. Keep version A for returning and VIP customers, "
                                   "where the free-shipping cost buys no extra orders."},
}
(HERE / "results.json").write_text(json.dumps(results, indent=2))

# ---------- Charts ----------
out = HERE / "charts"
fig, ax = plt.subplots(figsize=(8, 3.8))
xs = np.linspace(min(postA.min(), postB.min()), max(postA.max(), postB.max()), 400)
for post, lab, col in [(postA, "A: current email", "#8F9BBC"), (postB, "B: personalized + free shipping", S.PEACH)]:
    kde = stats.gaussian_kde(post)
    ax.fill_between(xs * 100, kde(xs), alpha=0.35, color=col)
    ax.plot(xs * 100, kde(xs), color=col, lw=2, label=lab)
ax.set_yticks([]); ax.grid(False)
ax.set_xlabel("Conversion rate (%)")
ax.set_title(f"B beats A with {prob_b_better:.1%} probability")
ax.legend(loc="upper left")
fig.savefig(out / "posterior.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.6))
segs_plot = segs.set_index("segment").loc[["New", "Returning", "VIP"]]
yy = np.arange(len(segs_plot))
for i, (s, row) in enumerate(segs_plot.iterrows()):
    col = S.MINT if row.significant_after_correction else S.MUTED
    ax.plot([row.lo * 100, row.hi * 100], [i, i], color=col, lw=3, solid_capstyle="round")
    ax.scatter([row.rel_lift * 100], [i], color=col, s=60, zorder=3)
    ax.text(row.hi * 100 + 1.5, i, f"{row.rel_lift:+.0%}" + ("" if row.significant_after_correction else "  (not significant)"),
            va="center", fontsize=9, color=S.INK)
ax.axvline(0, color=S.INK, lw=1)
ax.set_yticks(yy, segs_plot.index)
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Relative lift in conversion, B vs A (95% interval)")
ax.set_title("The lift comes from new customers")
lo_all = segs_plot.lo.min() * 100; hi_all = segs_plot.hi.max() * 100
ax.set_xlim(lo_all - 5, hi_all + 35)
fig.savefig(out / "segment_lift.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(boot * 100, bins=50, color="#B9C2DA", alpha=0.9, label="Everyone")
ax.hist(boot_new * 100, bins=50, color=S.MINT, alpha=0.75, label="New customers only")
ax.axvline(0, color=S.INK, lw=1)
ax.set_xlabel("Change in revenue per email sent, B minus A (cents)")
ax.set_yticks([]); ax.grid(False)
ax.set_title("Revenue per email: only the new-customer gain is clearly above zero")
ax.legend(loc="upper left")
fig.savefig(out / "revenue_bootstrap.png"); plt.close(fig)

print(json.dumps(results, indent=2, default=float))
