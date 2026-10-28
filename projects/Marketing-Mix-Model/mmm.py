"""
Marketing Mix Model and budget optimizer.

Question: a DTC brand spends about $2M a year across five channels.
Which channels actually drive revenue, and how should next year's budget move?

Steps
1. Simulate three years of weekly spend and revenue with known channel effects
   (carryover and diminishing returns), so the model can be checked against the truth.
2. Transform spend with geometric adstock and a Hill saturation curve.
3. Fit all channels jointly with non-negative effects, seasonality and price.
   Weak priors on carryover (what is typical for each channel type) keep the
   model stable, the same idea Bayesian MMM tools use. Validate on the last 16 weeks.
4. Report ROI and marginal ROI by channel.
5. Reallocate the same total budget to maximize predicted revenue.

Run: python mmm.py   (writes charts/ and results.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import _style as S  # noqa: E402

S.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

rng = np.random.default_rng(42)
WEEKS = 156
CHANNELS = ["Paid Search", "Paid Social", "TV", "Email", "Affiliate"]

# ---------- 1. Simulated ground truth ----------
TRUE = {  # carryover, half-saturation (weekly $), max weekly revenue
    "Paid Search": (0.20, 9_000, 70_000),
    "Paid Social": (0.40, 12_000, 60_000),
    "TV":          (0.70, 30_000, 85_000),
    "Email":       (0.10, 1_500, 22_000),
    "Affiliate":   (0.30, 4_000, 18_000),
}
BASE_SPEND = {"Paid Search": 11_000, "Paid Social": 9_000, "TV": 14_000, "Email": 1_200, "Affiliate": 3_000}

t = np.arange(WEEKS)
spend = {}
for ch, base in BASE_SPEND.items():
    noise = rng.lognormal(0, 0.35, WEEKS)
    pulse = 1 + 0.5 * (np.sin(2 * np.pi * (t - 40) / 52) > 0.6)  # holiday push
    s = base * noise * pulse
    if ch == "TV":
        s = s * (rng.random(WEEKS) > 0.35)  # TV runs in flights
    if ch in ("Email", "Affiliate"):
        s = s * (rng.random(WEEKS) > 0.25)  # some weeks have no sends or no partner promos
    spend[ch] = s
df = pd.DataFrame(spend)
df.insert(0, "week", pd.date_range("2023-10-02", periods=WEEKS, freq="W-MON"))
season = np.sin(2 * np.pi * t / 52 - 1.2)
price_idx = 1 + rng.normal(0, 0.03, WEEKS)


def adstock(x, a):
    out = np.zeros_like(x, dtype=float)
    for i, v in enumerate(x):
        out[i] = v + (a * out[i - 1] if i else 0)
    return out


def hill(x, k, shape=1.4):
    return x**shape / (x**shape + k**shape)


true_contrib = {ch: TRUE[ch][2] * hill(adstock(df[ch].values, TRUE[ch][0]), TRUE[ch][1]) for ch in CHANNELS}
baseline = 150_000 + 25_000 * season - 120_000 * (price_idx - 1)
df["revenue"] = baseline + sum(true_contrib.values()) + rng.normal(0, 6_000, WEEKS)

# Add the kind of mess real exports have, then clean it.
messy = df.copy()
messy.loc[rng.choice(WEEKS, 3, replace=False), "Email"] = np.nan  # missing rows from the ESP export
messy.loc[rng.choice(WEEKS, 2, replace=False), "Affiliate"] *= -1  # refunds booked as negative spend
messy = pd.concat([messy, messy.iloc[[10, 55]]])  # duplicated weeks
cleaning = {
    "duplicate_weeks_removed": int(messy.duplicated("week").sum()),
    "missing_email_spend_filled": int(messy["Email"].isna().sum()),
    "negative_spend_rows_fixed": int((messy[CHANNELS] < 0).sum().sum()),
}
clean = messy.drop_duplicates("week").sort_values("week").reset_index(drop=True)
clean["Email"] = clean["Email"].interpolate()
clean[CHANNELS] = clean[CHANNELS].abs()
df = clean
df.to_csv(HERE / "data" / "weekly_marketing.csv", index=False)

# ---------- 2-3. Fit ----------
TRAIN = WEEKS - 16
controls = np.column_stack([np.ones(WEEKS), season, price_idx - 1])


def design(params):
    cols = []
    for ch in CHANNELS:
        a, k = params[ch]
        cols.append(hill(adstock(df[ch].values, a), k))
    return np.column_stack(cols)


def fit(params, rows):
    X = design(params)[rows]
    # media coefficients must be >= 0; let controls take any sign by splitting them
    C = controls[rows]
    A = np.column_stack([X, C, -C])
    coef, _ = nnls(A, df["revenue"].values[rows])
    media = coef[: len(CHANNELS)]
    ctrl = coef[len(CHANNELS): len(CHANNELS) + 3] - coef[len(CHANNELS) + 3:]
    return media, ctrl


def predict(params, media, ctrl, rows=slice(None)):
    return design(params)[rows] @ media + controls[rows] @ ctrl


# joint fit: carryover a and saturation k for every channel, with weak priors on a
from scipy.optimize import least_squares  # noqa: E402

PRIOR_A = {"Paid Search": 0.2, "Paid Social": 0.3, "TV": 0.6, "Email": 0.2, "Affiliate": 0.2}
PRIOR_W = 25_000  # strength of the carryover prior, in revenue units
means = np.array([df[ch].mean() for ch in CHANNELS])
train_rows = np.arange(TRAIN)


def unpack(theta):
    n = len(CHANNELS)
    return {ch: (theta[i], np.exp(theta[n + i]) * means[i]) for i, ch in enumerate(CHANNELS)}


def residuals(theta):
    p = unpack(theta)
    m, c = fit(p, train_rows)
    r = predict(p, m, c, train_rows) - df["revenue"].values[train_rows]
    prior = [PRIOR_W * (p[ch][0] - PRIOR_A[ch]) for ch in CHANNELS]
    return np.concatenate([r, prior])


best = None
for start in range(6):
    r0 = np.random.default_rng(start)
    x0 = np.concatenate([[PRIOR_A[c] for c in CHANNELS], r0.uniform(-0.3, 1.0, len(CHANNELS))])
    sol = least_squares(residuals, x0,
                        bounds=([0.0] * len(CHANNELS) + [np.log(0.3)] * len(CHANNELS),
                                [0.9] * len(CHANNELS) + [np.log(6.0)] * len(CHANNELS)))
    if best is None or sol.cost < best.cost:
        best = sol
params = unpack(best.x)

media, ctrl = fit(params, train_rows)
pred = predict(params, media, ctrl)
y = df["revenue"].values
test = slice(TRAIN, WEEKS)
mape_test = float(np.mean(np.abs((pred[test] - y[test]) / y[test])))
r2_train = float(1 - np.sum((pred[:TRAIN] - y[:TRAIN]) ** 2) / np.sum((y[:TRAIN] - y[:TRAIN].mean()) ** 2))

# ---------- 4. ROI ----------
X = design(params)
contrib = {ch: X[:, i] * media[i] for i, ch in enumerate(CHANNELS)}
roi = {}
for i, ch in enumerate(CHANNELS):
    total_spend = df[ch].sum()
    a, k = params[ch]
    bumped = df[ch].values * 1.01
    marg = (media[i] * hill(adstock(bumped, a), k)).sum() - contrib[ch].sum()
    roi[ch] = {
        "spend": round(total_spend),
        "attributed_revenue": round(contrib[ch].sum()),
        "roi": round(contrib[ch].sum() / total_spend, 2),
        "marginal_roi": round(marg / (0.01 * total_spend), 2),
        "true_roi": round(true_contrib[ch].sum() / total_spend, 2),
        "carryover": round(float(a), 2),
    }

# ---------- 5. Budget optimizer (same total, +-50% per channel) ----------
weekly_budget = np.array([df[ch].mean() for ch in CHANNELS])


def media_rev(w):
    total = 0
    for i, ch in enumerate(CHANNELS):
        a, k = params[ch]
        steady = w[i] / (1 - a)  # steady-state adstock of constant spend
        total += media[i] * hill(np.array([steady]), k)[0]
    return total


res = minimize(lambda w: -media_rev(w), weekly_budget,
               bounds=[(0.5 * b, 1.5 * b) for b in weekly_budget],
               constraints=[{"type": "eq", "fun": lambda w: w.sum() - weekly_budget.sum()}],
               method="SLSQP")
opt = res.x
lift_weekly = media_rev(opt) - media_rev(weekly_budget)
lift_annual = lift_weekly * 52
lift_pct_media = lift_weekly / media_rev(weekly_budget)

results = {
    "data_cleaning": cleaning,
    "fit": {"train_r2": round(r2_train, 3), "holdout_mape": round(mape_test, 3), "holdout_weeks": WEEKS - TRAIN},
    "channels": roi,
    "budget": {
        "annual_budget": round(weekly_budget.sum() * 52),
        "current_mix": {ch: round(weekly_budget[i] * 52) for i, ch in enumerate(CHANNELS)},
        "optimized_mix": {ch: round(opt[i] * 52) for i, ch in enumerate(CHANNELS)},
        "annual_revenue_lift": round(lift_annual),
        "media_revenue_lift_pct": round(float(lift_pct_media), 3),
    },
}
(HERE / "results.json").write_text(json.dumps(results, indent=2))

# ---------- Charts ----------
out = HERE / "charts"
fmt = FuncFormatter(S.money)

fig, ax = plt.subplots(figsize=(10, 4.2))
base_part = y - sum(contrib.values())
ax.stackplot(df["week"], [np.clip(base_part, 0, None)] + [contrib[c] for c in CHANNELS],
             labels=["Baseline"] + CHANNELS,
             colors=["#E7EAF3"] + S.PALETTE[: len(CHANNELS)], alpha=0.95)
ax.plot(df["week"], y, color=S.INK, lw=1.2, label="Actual revenue")
ax.axvline(df["week"].iloc[TRAIN], color=S.MUTED, ls="--", lw=1)
ax.text(df["week"].iloc[TRAIN], ax.get_ylim()[1] * 0.97, "  holdout", color=S.MUTED, va="top")
ax.yaxis.set_major_formatter(fmt)
ax.set_title("Weekly revenue, broken down by what drove it")
ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
fig.savefig(out / "decomposition.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
order = sorted(CHANNELS, key=lambda c: roi[c]["roi"])
yy = np.arange(len(order))
ax.barh(yy - 0.2, [roi[c]["roi"] for c in order], 0.38, color=S.COBALT, label="Average ROI")
ax.barh(yy + 0.2, [roi[c]["marginal_roi"] for c in order], 0.38, color=S.PEACH, label="Marginal ROI (next $1)")
ax.set_yticks(yy, order)
ax.axvline(1, color=S.INK, lw=1)
ax.text(1.02, len(order) - 0.45, "break-even", color=S.INK, fontsize=8.5)
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Revenue per $1 spent")
ax.set_title("Where the next dollar works hardest")
ax.legend(loc="lower right")
fig.savefig(out / "roi.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
xx = np.arange(len(CHANNELS))
ax.bar(xx - 0.2, weekly_budget * 52, 0.38, color="#B9C2DA", label="Current")
ax.bar(xx + 0.2, opt * 52, 0.38, color=S.MINT, label="Recommended")
ax.set_xticks(xx, CHANNELS)
ax.yaxis.set_major_formatter(fmt)
ax.set_title(f"Same budget, re-mixed: +{S.money(lift_annual)} revenue a year")
ax.legend()
fig.savefig(out / "budget_shift.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
for i, ch in enumerate(CHANNELS):
    a, k = params[ch]
    grid = np.linspace(0, weekly_budget[i] * 3, 200)
    curve = media[i] * hill(grid / (1 - a), k)
    ax.plot(grid, curve, label=ch, lw=2)
    ax.scatter([weekly_budget[i]], [media[i] * hill(np.array([weekly_budget[i] / (1 - a)]), k)[0]], s=28, zorder=3)
ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
ax.set_xlabel("Weekly spend"); ax.set_ylabel("Weekly revenue from channel")
ax.set_title("Response curves: dots show where each channel sits today")
ax.legend(fontsize=8.5)
fig.savefig(out / "response_curves.png")
plt.close(fig)

print(json.dumps(results, indent=2))
