"""
Kaplan-Meier survival analysis on peace agreement duration (span_days).

Event indicator: ended == 1  (agreement collapsed / terminated)
Censored:        ended is NaN or 0  (agreement ongoing at censor point)

Nine plots produced in a 5-row layout:
  Row 1: Overall | By incompatibility type
  Row 2: By agreement type | By region
  Row 3: By shaloc (local/regional autonomy provision) | By PKO presence
  Row 4: By intgov (integration of rebels into government provision)  [full width]
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import chi2

# ── KM estimator ──────────────────────────────────────────────────────────────

def km_curve(durations, events):
    """Return (times, survival, lower_95, upper_95) using Greenwood's formula."""
    order = np.argsort(durations)
    t = np.asarray(durations, float)[order]
    e = np.asarray(events,    float)[order]

    unique_times, d_list, n_list = [], [], []
    n = len(t)
    i = 0
    while i < len(t):
        ti = t[i]
        j  = i
        while j < len(t) and t[j] == ti:
            j += 1
        deaths = e[i:j].sum()
        if deaths > 0:
            unique_times.append(ti)
            d_list.append(deaths)
            n_list.append(n - i)
        i = j

    times  = np.array(unique_times, float)
    d      = np.array(d_list,       float)
    n_risk = np.array(n_list,       float)

    S          = np.cumprod(1 - d / n_risk)
    # Greenwood is undefined when a death empties the risk set. That term is
    # omitted; adding a tiny number in the denominator would explode the band.
    denom      = n_risk * (n_risk - d)
    incr       = np.zeros_like(d)
    ok         = denom > 0
    incr[ok]   = d[ok] / denom[ok]
    greenwood  = np.cumsum(incr)
    s_clip     = np.clip(S, 1e-12, 1 - 1e-12)
    log_log_S  = np.log(-np.log(s_clip))
    var_ll     = greenwood / np.log(s_clip) ** 2
    z          = 1.96
    lower      = np.exp(-np.exp(log_log_S + z * np.sqrt(var_ll)))
    upper      = np.exp(-np.exp(log_log_S - z * np.sqrt(var_ll)))

    times = np.concatenate([[0], times])
    S     = np.concatenate([[1], S])
    lower = np.concatenate([[1], lower])
    upper = np.concatenate([[1], upper])
    return times, S, lower, upper


# ── log-rank test ─────────────────────────────────────────────────────────────

def log_rank_test(groups):
    """Multivariate log-rank test. Returns a chi-square p-value on k-1 df.

    The covariance is the hypergeometric variance of the death counts, so the
    statistic accounts for the dependence among groups. Summing (O-E)^2/E over
    every group would treat those groups as independent.
    """
    groups = [(np.asarray(t, float), np.asarray(e, float)) for t, e in groups]
    event_times = [t[e == 1] for t, e in groups if np.any(e == 1)]
    if len(groups) < 2 or not event_times:
        return np.nan
    event_times = np.unique(np.concatenate(event_times))
    k = len(groups)
    observed = np.zeros(k)
    expected = np.zeros(k)
    variance = np.zeros((k, k))
    for ti in event_times:
        at_risk = np.array([(t >= ti).sum() for t, _e in groups], float)
        deaths = np.array([((t == ti) & (e == 1)).sum() for t, e in groups], float)
        n = at_risk.sum()
        d = deaths.sum()
        if n <= 1 or d == 0:
            continue
        observed += deaths
        expected += at_risk * d / n
        factor = d * (n - d) / (n ** 2 * (n - 1))
        for a in range(k):
            variance[a, a] += factor * at_risk[a] * (n - at_risk[a])
            for b in range(a + 1, k):
                cov = factor * at_risk[a] * at_risk[b]
                variance[a, b] -= cov
                variance[b, a] -= cov
    diff = (observed - expected)[:-1]
    vv = variance[:-1, :-1]
    try:
        stat = float(diff @ np.linalg.solve(vv, diff))
    except np.linalg.LinAlgError:
        return np.nan
    return float(chi2.sf(stat, df=k - 1))


# ── helpers ───────────────────────────────────────────────────────────────────

def step_plot(ax, times, S, lower, upper, color, label, lw=2.2):
    ax.step(times / 365.25, S, where="post", color=color, lw=lw, label=label)
    ax.fill_between(times / 365.25, lower, upper,
                    step="post", alpha=0.13, color=color)


def style_ax(ax, title, xlabel="Time (years)", ylabel="Survival probability"):
    ax.set_title(title, fontsize=12.5, fontweight="bold",
                 pad=10, color="#1a1a1a")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.5, color="#bbb", lw=0.9, linestyle=":", zorder=0)


def add_pval(ax, p):
    label = "Log-rank  p < 0.001" if p < 0.001 else f"Log-rank  p = {p:.3f}"
    ax.text(0.03, 0.06, label,
            transform=ax.transAxes, fontsize=9,
            color="#444", style="italic")


def binary_km(ax, df, col, labels, colors, title):
    """Plot KM curves for a binary (0/1) variable and return log-rank p."""
    groups = []
    for i, (code, label) in enumerate(labels.items()):
        sub = df[df[col] == code]
        t_g, e_g = sub["t"].values, sub["event"].values
        groups.append((t_g, e_g))
        times, S, lo, hi = km_curve(t_g, e_g)
        step_plot(ax, times, S, lo, hi, colors[i],
                  f"{label}  (n={len(sub)}, ev={int(e_g.sum())})")
    p = log_rank_test(groups)
    style_ax(ax, title)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85, edgecolor="#ccc")
    add_pval(ax, p)
    return p


# ── load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv("pad_merged.csv")
df["censor_point"] = pd.to_datetime(df["censor_point"])
df["pa_date"]      = pd.to_datetime(df["pa_date"])

censored_mask = df["ended"].isna() | (df["ended"] != 1)
df.loc[censored_mask, "span_days"] = (
    df.loc[censored_mask, "censor_point"]
    - df.loc[censored_mask, "pa_date"]
).dt.days

df["ended"] = df["ended"].fillna(0)
df = df.dropna(subset=["span_days"]).copy()

df["event"] = (df["ended"] == 1).astype(int)
df["t"]     = df["span_days"].astype(float)
bad_time = ~np.isfinite(df["t"]) | (df["t"] <= 0)
if bad_time.any():
    print(f"  [INFO] Dropped {int(bad_time.sum())} rows with nonpositive duration")
    df = df.loc[~bad_time].copy()


# ── Derive prior conflict outcome from merged variables ────────────────────
# 0 = neither (ceasefire, low activity, unknown)
# 1 = government victory
# 2 = rebel victory
df["rebel_victory"]      = df["rebel_victory"].fillna(0).astype(int)
df["government_victory"] = df["government_victory"].fillna(0).astype(int)
df["prior_outcome"] = np.where(
    df["rebel_victory"]      == 1, 2,
    np.where(df["government_victory"] == 1, 1, 0)
)
# neighbor_at_war: fill island states (already 0 in merged data) 
df["neighbor_at_war"] = df["neighbor_at_war"].fillna(0).astype(int)

# ── label maps ────────────────────────────────────────────────────────────────

incomp_labels = {1: "Territory", 2: "Government", 3: "Both"}
patype_labels = {1: "Full",   2: "Partial", 3: "Peace Process"}
region_labels = {1: "Europe",    2: "Middle East",   3: "Asia",
                  4: "Africa",   5: "Americas"}


nbr_war_labels   = {0: "No neighbor at war", 1: "Neighbor at war"}
prior_out_labels = {0: "Other / unknown", 1: "Government victory", 2: "Rebel victory"}

# ── colour palette ────────────────────────────────────────────────────────────

PALETTE = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261", "#6A4C93"]

# ── figure ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Georgia", "Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e0dcd5",
    "grid.linewidth":    0.6,
    "grid.linestyle":    "--",
    "figure.facecolor":  "#FAFAF7",
    "axes.facecolor":    "#FAFAF7",
    "axes.labelcolor":   "#2b2b2b",
    "xtick.color":       "#2b2b2b",
    "ytick.color":       "#2b2b2b",
    "text.color":        "#2b2b2b",
})

fig = plt.figure(figsize=(16, 32), facecolor="#FAFAF7")
fig.suptitle(
    "Kaplan-Meier Survival Analysis of Peace Agreement Duration",
    fontsize=18, fontweight="bold", y=0.995, color="#1a1a1a",
)
fig.text(0.5, 0.988,
         "Outcome: agreement collapse (ended = 1)  |  Censored: ongoing agreements  |  Time in years",
         ha="center", fontsize=9.5, color="#666", style="italic")

axes = fig.subplot_mosaic(
    [["overall",       "incomp"],
     ["patype",        "region"],
     ["shaloc",        "pko"],
     ["intgov",        "nbr_war"],
     ["prior_outcome", "."]],
    gridspec_kw={"hspace": 0.40, "wspace": 0.28},
)

# ── 1. Overall ────────────────────────────────────────────────────────────────

ax = axes["overall"]
t, e = df["t"].values, df["event"].values
times, S, lo, hi = km_curve(t, e)
step_plot(ax, times, S, lo, hi, PALETTE[0],
          f"All agreements  (n={len(df)}, events={int(e.sum())})")
style_ax(ax, "Overall Survival")
ax.legend(fontsize=9, loc="upper right", framealpha=0.85, edgecolor="#ccc")

med_idx = np.searchsorted(-S, -0.5)
if med_idx < len(times):
    med_yrs = times[med_idx] / 365.25
    ax.axvline(med_yrs, color=PALETTE[0], lw=1.1, linestyle="--", alpha=0.55)
    ax.text(med_yrs + 0.3, 0.53, f"Median ~{med_yrs:.1f} yr",
            color=PALETTE[0], fontsize=8.5)

# ── 2. By incompatibility type ────────────────────────────────────────────────

ax = axes["incomp"]
groups_inc = []
for i, (code, label) in enumerate(incomp_labels.items()):
    sub = df[df["incompatibility"] == code]
    if len(sub) < 5:
        continue
    t_g, e_g = sub["t"].values, sub["event"].values
    groups_inc.append((t_g, e_g))
    times, S, lo, hi = km_curve(t_g, e_g)
    step_plot(ax, times, S, lo, hi, PALETTE[i],
              f"{label}  (n={len(sub)}, ev={int(e_g.sum())})")
p_inc = log_rank_test(groups_inc)
style_ax(ax, "By Incompatibility Type")
ax.legend(fontsize=9, loc="upper right", framealpha=0.85, edgecolor="#ccc")
add_pval(ax, p_inc)

# ── 3. By agreement type ─────────────────────────────────────────────────────

ax = axes["patype"]
groups_pa = []
for i, (code, label) in enumerate(patype_labels.items()):
    sub = df[df["pa_type"] == code]
    if len(sub) < 5:
        continue
    t_g, e_g = sub["t"].values, sub["event"].values
    groups_pa.append((t_g, e_g))
    times, S, lo, hi = km_curve(t_g, e_g)
    step_plot(ax, times, S, lo, hi, PALETTE[i],
              f"{label}  (n={len(sub)}, ev={int(e_g.sum())})")
p_pa = log_rank_test(groups_pa)
style_ax(ax, "By Agreement Type")
ax.legend(fontsize=9, loc="upper right", framealpha=0.85, edgecolor="#ccc")
add_pval(ax, p_pa)

# ── 4. By region ──────────────────────────────────────────────────────────────

ax = axes["region"]
groups_reg = []
for i, (code, label) in enumerate(region_labels.items()):
    sub = df[df["region"] == code]
    if len(sub) < 5:
        continue
    t_g, e_g = sub["t"].values, sub["event"].values
    groups_reg.append((t_g, e_g))
    times, S, lo, hi = km_curve(t_g, e_g)
    step_plot(ax, times, S, lo, hi, PALETTE[i % len(PALETTE)],
              f"{label}  (n={len(sub)}, ev={int(e_g.sum())})")
p_reg = log_rank_test(groups_reg)
style_ax(ax, "By Region")
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85, edgecolor="#ccc")
add_pval(ax, p_reg)

# ── 5. By shaloc (local/regional autonomy provision) ─────────────────────────

binary_km(
    axes["shaloc"], df, "shaloc",
    labels={0: "No local power-sharing provision", 1: "Local power-sharing provision included"},
    colors=[PALETTE[3], PALETTE[0]],
    title="By Local Power-Sharing Provision (shaloc)",
)

# ── 6. By PKO presence ────────────────────────────────────────────────────────

binary_km(
    axes["pko"], df, "pko",
    labels={0: "No peacekeeping operation", 1: "PKO present"},
    colors=[PALETTE[2], PALETTE[1]],
    title="By Peacekeeping Operation Presence (pko)",
)

# ── 7. By intgov (integration into government provision) ────────────────────

binary_km(
    axes["intgov"], df, "intgov",
    labels={0: "No integration into government provision", 1: "Integration into government included"},
    colors=[PALETTE[5], PALETTE[4]],
    title="By Integration into Government Provision (intgov)",
)



# ── 8. By neighbor war status ─────────────────────────────────────────────────

binary_km(
    axes["nbr_war"], df, "neighbor_at_war",
    labels=nbr_war_labels,
    colors=[PALETTE[2], PALETTE[0]],
    title="By Neighbor War Status (neighbor_at_war)",
)

# ── 9. By prior conflict outcome (3 categories) ───────────────────────────────

ax = axes["prior_outcome"]
groups_po = []
for i, (code, label) in enumerate(prior_out_labels.items()):
    sub = df[df["prior_outcome"] == code]
    if len(sub) < 5:
        continue
    t_g, e_g = sub["t"].values, sub["event"].values
    groups_po.append((t_g, e_g))
    times, S, lo, hi = km_curve(t_g, e_g)
    step_plot(ax, times, S, lo, hi, PALETTE[i % len(PALETTE)],
              f"{label}  (n={len(sub)}, ev={int(e_g.sum())})")
p_po = log_rank_test(groups_po)
style_ax(ax, "By Prior Conflict Outcome")
ax.legend(fontsize=9, loc="upper right", framealpha=0.85, edgecolor="#ccc")
add_pval(ax, p_po)

# ── save ──────────────────────────────────────────────────────────────────────

out_png = "kaplan_meier_full.png"
plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="#FAFAF7")
print(f"Saved to {out_png}")
plt.close()