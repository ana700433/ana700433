"""
Multi-touch attribution: rule-based models vs a data-driven Markov chain model.

Question: last-click reporting says Paid Search drives most sales.
Is that true, or is Search just the last stop on a journey other channels started?

Steps
1. Simulate 60,000 customer journeys across 6 channels. Upper-funnel channels
   (Display, Social, Video) start journeys; Search and Direct tend to close them.
2. Clean the raw touchpoint log: drop bot sessions, collapse repeat clicks,
   and cap journeys at a 30-day lookback.
3. Credit conversions with first-touch, last-touch, linear, time-decay,
   position-based (40/20/40) and a Markov chain removal-effect model.
4. Compare how much credit each channel gets under each model.

Run: python attribution.py   (writes charts/ and results.json)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import _style as S  # noqa: E402

S.apply()
import matplotlib.pyplot as plt  # noqa: E402

rng = np.random.default_rng(7)
CH = ["Display", "Paid Social", "Video", "Email", "Paid Search", "Direct"]
N = 60_000

# ---------- 1. Simulate journeys ----------
start_p = np.array([0.26, 0.30, 0.14, 0.08, 0.17, 0.05])
# next-step probabilities: upper funnel hands off to Search, Email and Direct
T = np.array([
    #  Disp  Soc   Vid   Email Srch  Dir
    [0.10, 0.12, 0.06, 0.10, 0.30, 0.12],  # Display
    [0.08, 0.14, 0.08, 0.12, 0.28, 0.12],  # Paid Social
    [0.06, 0.14, 0.08, 0.10, 0.30, 0.14],  # Video
    [0.04, 0.06, 0.02, 0.10, 0.22, 0.26],  # Email
    [0.03, 0.05, 0.02, 0.10, 0.12, 0.20],  # Paid Search
    [0.02, 0.03, 0.01, 0.06, 0.10, 0.10],  # Direct
])
conv_p = np.array([0.004, 0.006, 0.006, 0.030, 0.055, 0.085])  # chance a step ends in a purchase
stop = 1 - T.sum(1) - conv_p  # chance the journey just ends

rows = []
for uid in range(N):
    c = rng.choice(6, p=start_p)
    day = 0.0
    while True:
        rows.append((uid, CH[c], day, len(rows)))
        r = rng.random()
        if r < conv_p[c]:
            rows.append((uid, "CONVERSION", day, len(rows))); break
        if r < conv_p[c] + stop[c]:
            break
        c = rng.choice(6, p=T[c] / T[c].sum())
        day += rng.exponential(3)
log = pd.DataFrame(rows, columns=["user_id", "event", "day", "seq"])

# realistic noise: bots, double clicks, very old touches
bots = rng.choice(N, 900, replace=False)
dup = log.sample(4_000, random_state=1)
log = pd.concat([log, dup]).sort_values(["user_id", "seq"], kind="mergesort").reset_index(drop=True)
log["is_bot"] = log["user_id"].isin(bots)
raw_rows = len(log)

# ---------- 2. Clean ----------
clean = log[~log["is_bot"]].copy()
removed_bots = raw_rows - len(clean)
before = len(clean)
clean = clean.loc[~(clean["event"].eq(clean.groupby("user_id")["event"].shift()) &
                    clean["day"].eq(clean.groupby("user_id")["day"].shift()))]
removed_dupes = before - len(clean)
end_day = clean.groupby("user_id")["day"].transform("max")
before = len(clean)
clean = clean[end_day - clean["day"] <= 30]
removed_old = before - len(clean)
clean.to_csv(HERE / "data" / "touchpoints_clean.csv.gz", index=False, compression="gzip")

paths = clean.groupby("user_id")["event"].apply(list)
conv_paths = [p[:-1] for p in paths if p[-1] == "CONVERSION" and len(p) > 1]
lost_paths = [p for p in paths if p[-1] != "CONVERSION"]
days = clean.groupby("user_id")["day"].apply(list)
conv_days = [d[:-1] for p, d in zip(paths, days) if p[-1] == "CONVERSION" and len(p) > 1]
total_conv = len(conv_paths)

# ---------- 3. Models ----------
def credit(rule):
    out = defaultdict(float)
    for p, d in zip(conv_paths, conv_days):
        n = len(p)
        if rule == "First touch":
            w = [1] + [0] * (n - 1)
        elif rule == "Last touch":
            w = [0] * (n - 1) + [1]
        elif rule == "Linear":
            w = [1 / n] * n
        elif rule == "Time decay":
            end = d[-1]
            raw = [0.5 ** ((end - x) / 7) for x in d]  # 7-day half-life
            w = [r / sum(raw) for r in raw]
        else:  # Position based 40/20/40
            if n == 1:
                w = [1]
            elif n == 2:
                w = [0.5, 0.5]
            else:
                w = [0.4] + [0.2 / (n - 2)] * (n - 2) + [0.4]
        for ch, x in zip(p, w):
            out[ch] += x
    return out


def markov():
    counts = defaultdict(lambda: defaultdict(int))
    for p in conv_paths:
        seq = ["START"] + p + ["CONV"]
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
    for p in lost_paths:
        seq = ["START"] + p + ["NULL"]
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
    states = ["START"] + CH + ["CONV", "NULL"]
    idx = {s: i for i, s in enumerate(states)}
    M = np.zeros((len(states), len(states)))
    for a, nxt in counts.items():
        tot = sum(nxt.values())
        for b, c in nxt.items():
            M[idx[a], idx[b]] = c / tot
    M[idx["CONV"], idx["CONV"]] = 1
    M[idx["NULL"], idx["NULL"]] = 1

    def p_conv(Mx):
        transient = [idx[s] for s in ["START"] + CH]
        Q = Mx[np.ix_(transient, transient)]
        R = Mx[np.ix_(transient, [idx["CONV"]])]
        Nf = np.linalg.solve(np.eye(len(transient)) - Q, R)
        return Nf[0, 0]

    base = p_conv(M)
    removal = {}
    for ch in CH:
        Mx = M.copy()
        i = idx[ch]
        Mx[:, idx["NULL"]] += Mx[:, i]
        Mx[:, i] = 0
        Mx[i, :] = 0
        Mx[i, idx["NULL"]] = 1
        removal[ch] = 1 - p_conv(Mx) / base
    tot = sum(removal.values())
    return {ch: removal[ch] / tot * total_conv for ch in CH}, removal, base


MODELS = ["Last touch", "First touch", "Linear", "Time decay", "Position based"]
table = {m: credit(m) for m in MODELS}
mk, removal, base_rate = markov()
table["Markov chain"] = mk
share = pd.DataFrame({m: {ch: table[m].get(ch, 0) / total_conv for ch in CH} for m in table}).loc[CH]

journey_len = pd.Series([len(p) for p in conv_paths])
results = {
    "cleaning": {"raw_rows": raw_rows, "bot_rows_removed": int(removed_bots),
                 "duplicate_clicks_removed": int(removed_dupes), "touches_outside_30_days_removed": int(removed_old)},
    "journeys": {"users": int(len(paths)), "conversions": total_conv,
                 "conversion_rate": round(total_conv / len(paths), 4),
                 "median_touches_before_purchase": float(journey_len.median()),
                 "share_multi_touch": round(float((journey_len > 1).mean()), 3)},
    "credit_share": share.round(3).to_dict(),
    "markov_removal_effect": {k: round(v, 3) for k, v in removal.items()},
}
(HERE / "results.json").write_text(json.dumps(results, indent=2))

# ---------- 4. Charts ----------
out = HERE / "charts"
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(CH)); w = 0.13
colors = ["#B9C2DA", "#8F9BBC", "#C9CFE0", "#A5B0CC", "#7E8AAE", S.PEACH]
for i, m in enumerate(share.columns):
    ax.bar(x + (i - 2.5) * w, share[m] * 100, w, label=m, color=colors[i])
ax.set_xticks(x, CH)
ax.set_ylabel("% of conversions credited")
ax.set_title("Same sales, six stories: credit by channel under each model")
ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8.5)
fig.savefig(out / "model_comparison.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
diff = (share["Markov chain"] - share["Last touch"]) * 100
order = diff.sort_values().index
ax.barh(order, diff[order], color=[S.MINT if v > 0 else S.ROSE for v in diff[order]])
ax.axvline(0, color=S.INK, lw=1)
for i, ch in enumerate(order):
    v = diff[ch]
    ax.text(v + (0.3 if v > 0 else -0.3), i, f"{v:+.1f} pts", va="center", ha="left" if v > 0 else "right", fontsize=9, color=S.INK)
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Change in credit share, Markov vs last touch (percentage points)")
ax.set_title("Who last-click reporting undervalues")
pad = max(abs(diff.min()), diff.max()) * 1.35
ax.set_xlim(-pad, pad)
fig.savefig(out / "markov_vs_last_touch.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.6))
vc = journey_len.clip(upper=6).value_counts().sort_index()
ax.bar([str(i) if i < 6 else "6+" for i in vc.index], vc.values / vc.sum() * 100, color=S.COBALT)
ax.set_xlabel("Touchpoints before purchase"); ax.set_ylabel("% of buyers")
ax.set_title("Most buyers touch more than one channel")
fig.savefig(out / "journey_length.png"); plt.close(fig)

print(json.dumps(results, indent=2))
