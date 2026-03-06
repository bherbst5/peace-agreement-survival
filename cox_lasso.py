"""
Lasso-Penalized Cox Proportional Hazards Model  —  scikit-survival version
===========================================================================
Direct comparison counterpart to the custom FISTA implementation.

Key design choices that mirror the custom code:
  • Same data prep, dummy encoding, interaction terms
  • CoxnetSurvivalAnalysis with l1_ratio=1.0  (pure LASSO)
  • normalize=False  — we standardise manually (mean 0, sd 1) so the
    λ grid and back-transformation are directly comparable
  • λ_max computed from gradient at β=0, identical to custom code
  • Same 5-fold CV criterion: mean held-out partial log-likelihood
  • Bootstrap 95% CIs at λ_min (B=200, same fixed-λ approach)
  • Same 4-panel figure layout and visual style

Scaling note
------------
scikit-survival scales the loss by 1/n_events  (matching the custom code),
NOT by 1/n as R's glmnet does.  This means λ values ARE numerically
comparable between this script and the custom FISTA version.

Requirements
------------
    pip install scikit-survival openpyxl matplotlib pandas numpy
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA PREP
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_excel("~/thesis/surv_modeling/PAD_final.xlsx")
df = df.rename(columns={"ref ": "ref", "return ": "return"})

# ── Compute span_days for censored rows using censor_point ───────────────────
# ended = NaN or 0  →  censored; use censor_point as the end date
df["censor_point"] = pd.to_datetime(df["censor_point"])
df["pa_date"]    = pd.to_datetime(df["pa_date"])

censored_mask = df["ended"].isna() | (df["ended"] != 1)
df.loc[censored_mask, "span_days"] = (
    df.loc[censored_mask, "censor_point"]
    - df.loc[censored_mask, "pa_date"]
).dt.days

df["ended"] = df["ended"].fillna(0)

# ── Drop rows where span_days is still missing (no start date available) ─────
n_before = len(df)
df = df.dropna(subset=["span_days"]).copy()
n_after = len(df)
if n_before != n_after:
    print(f"  [INFO] Dropped {n_before - n_after} rows with NaN in "
          f"span_days  ({n_after} rows remain)")

df["event"] = (df["ended"] == 1).astype(bool)   # sksurv needs bool
df["t"]     = df["span_days"].astype(float)

# ── Binary variables (28) ─────────────────────────────────────────────────────
BINARY = [
    "cease", "intarmy", "ddr", "withd", "pp", "intgov", "intciv",
    "elections", "interim", "natalks", "aut", "fed", "ind", "ref",
    "shaloc", "regdev", "cul", "demarcation", "locgov", "amn", "pris",
    "recon", "return", "reaffirm", "outlin", "pko", "gender", "co_impl",
]

# ── Ordinal / continuous ──────────────────────────────────────────────────────
ORDINAL = ["termdur"]

# ── Multi-level → dummy-encode ────────────────────────────────────────────────
for code, name in [(2, "gov"), (3, "both")]:
    df[f"incomp_{name}"] = (df["incompatibility"] == code).astype(float)
for code, name in [(2, "partial"), (3, "process")]:
    df[f"patype_{name}"] = (df["pa_type"] == code).astype(float)
for code, name in [(2, "frame2"), (3, "frame3")]:
    df[f"frame_{name}"] = (df["frame"] == code).astype(float)
for code, name in [(1, "europe"), (2, "middleeast"), (3, "asia"), (5, "americas")]:
    df[f"region_{name}"] = (df["region"] == code).astype(float)

DUMMIES = [
    "region_europe", "region_middleeast", "region_asia", "region_americas",
    "incomp_gov", "incomp_both",
    "patype_partial", "patype_process",
    "frame_frame2", "frame_frame3",
]
COVARIATE_NAMES = BINARY + ORDINAL + DUMMIES

INTERACTIONS = [
    ("cease",       "pko"),
    ("ddr",         "intarmy"),
    ("ddr",         "intgov"),
    ("pp",          "elections"),
    ("amn",         "co_impl"),
    ("shaloc",      "return"),
    ("aut",         "shaloc"),
    ("withd",       "pko"),
    ("reaffirm",    "co_impl"),
    ("demarcation", "locgov"),
    ("pris",        "amn"),
    ("amn",         "recon"),
    ("intgov",      "natalks"),
]

for (a, b) in INTERACTIONS:
    assert a in COVARIATE_NAMES, f"Interaction term '{a}' not in COVARIATE_NAMES"
    assert b in COVARIATE_NAMES, f"Interaction term '{b}' not in COVARIATE_NAMES"

INTERACTION_NAMES = [f"{a}::{b}" for (a, b) in INTERACTIONS]

LABELS = {
    "region_europe":      "Region: Europe [ref: Africa]",
    "region_middleeast":  "Region: Middle East [ref: Africa]",
    "region_asia":        "Region: Asia [ref: Africa]",
    "region_americas":    "Region: Americas [ref: Africa]",
    "cease":              "Ceasefire provision",
    "intarmy":            "Integration of armed forces",
    "ddr":                "DDR provision",
    "withd":              "Withdrawal provision",
    "pp":                 "Power-sharing (political)",
    "intgov":             "Interim government",
    "intciv":             "Civilian power-sharing",
    "elections":          "Elections provision",
    "interim":            "Interim arrangements",
    "natalks":            "National dialogue / talks",
    "aut":                "Autonomy provision",
    "fed":                "Federalism provision",
    "ind":                "Independence provision",
    "ref":                "Referendum provision",
    "shaloc":             "Local power-sharing (shaloc)",
    "regdev":             "Regional development",
    "cul":                "Cultural provisions",
    "demarcation":        "Border demarcation",
    "locgov":             "Local governance",
    "amn":                "Amnesty provision",
    "pris":               "Prisoner release",
    "recon":              "Reconciliation",
    "return":             "Return / resettlement",
    "reaffirm":           "Reaffirmation of prev. agreement",
    "outlin":             "Outline agreement",
    "pko":                "Peacekeeping operation",
    "gender":             "Gender provision",
    "co_impl":            "Co-implementation body",
    "termdur":            "Term duration (termdur)",
    "incomp_gov":         "Incompatibility: Government [ref: Territory]",
    "incomp_both":        "Incompatibility: Both [ref: Territory]",
    "patype_partial":     "Agreement type: Partial [ref: Full]",
    "patype_process":     "Agreement type: Peace Process [ref: Full]",
    "frame_frame2":       "Frame type 2 [ref: Frame 1]",
    "frame_frame3":       "Frame type 3 [ref: Frame 1]",
}

t_all = df["t"].values.astype(float)
e_all = df["event"].values          # bool array for sksurv

# ── Standardise base variables first ─────────────────────────────────────────
X_base_raw = df[COVARIATE_NAMES].values.astype(float)
X_mean     = X_base_raw.mean(axis=0)
X_sd       = X_base_raw.std(axis=0)
X_sd[X_sd == 0] = 1.0
X_base_std = (X_base_raw - X_mean) / X_sd

# ── Build interactions via residualization ────────────────────────────────────
# Regress (x_a_std * x_b_std) on x_a_std and x_b_std and keep the residual.
# Guarantees orthogonality to both main effects regardless of variable type.
from numpy.linalg import lstsq

n = X_base_std.shape[0]
ia_resid = []
for a, b in INTERACTIONS:
    col_a    = X_base_std[:, COVARIATE_NAMES.index(a)]
    col_b    = X_base_std[:, COVARIATE_NAMES.index(b)]
    product  = col_a * col_b
    Z        = np.column_stack([np.ones(n), col_a, col_b])
    coefs, _, _, _ = lstsq(Z, product, rcond=None)
    ia_resid.append(product - Z @ coefs)

X_ia     = np.column_stack(ia_resid)
ia_mean  = X_ia.mean(axis=0)
ia_sd    = X_ia.std(axis=0)
ia_sd[ia_sd == 0] = 1.0
X_ia_std = (X_ia - ia_mean) / ia_sd

# ── Full 52-column standardised design matrix ─────────────────────────────────
X_std     = np.hstack([X_base_std, X_ia_std])
X_sd_full = np.concatenate([X_sd, ia_sd])   # used for all back-transformations

# Update COVARIATE_NAMES and LABELS to include interactions
COVARIATE_NAMES = COVARIATE_NAMES + INTERACTION_NAMES
for iname, (a, b) in zip(INTERACTION_NAMES, INTERACTIONS):
    LABELS[iname] = (f"{LABELS[a].split(' [ref')[0]}  x  "
                     f"{LABELS[b].split(' [ref')[0]}")

n, p = X_std.shape
print(f"  + {len(INTERACTIONS)} interaction term(s), residualised → "
      f"design matrix {p} columns")

# ═══════════════════════════════════════════════════════════════════════════════
# 1b.  PRE-FIT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  PRE-FIT DIAGNOSTIC CHECKS")
print("─" * 60)

e_float = e_all.astype(float)
sep_warnings = 0
print("\n[Check 1] Zero-variance / separation screening:")
for j, name in enumerate(COVARIATE_NAMES):
    col = X_std[:, j]
    if col.std() == 0:
        print(f"  [WARNING] '{name}' has zero variance.")
        sep_warnings += 1
        continue
    unique_vals = np.unique(col)
    if name not in INTERACTION_NAMES and set(unique_vals).issubset({0.0, 1.0}):
        ev1 = e_float[col == 1].sum() if (col == 1).any() else 0
        ev0 = e_float[col == 0].sum() if (col == 0).any() else 0
        if ev1 == 0 or ev0 == 0:
            print(f"  [WARNING] '{name}' may cause complete separation: "
                  f"events in group-1={int(ev1)}, group-0={int(ev0)}.")
            sep_warnings += 1
if sep_warnings == 0:
    print("  No separation or zero-variance issues detected.")

print(f"\nN={n}  Events={int(e_float.sum())}  "
      f"Censored={int((e_float == 0).sum())}")
print(f"Design matrix: {p} columns  "
      f"({len(BINARY)} binary, {len(ORDINAL)} ordinal, {len(DUMMIES)} dummies, "
      f"{len(INTERACTIONS)} interactions)")

# ── sksurv structured array ───────────────────────────────────────────────────
y_surv = Surv.from_arrays(event=e_all, time=t_all)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CORE HELPER: partial log-likelihood
#     (Breslow, scaled by 1/n_events — identical to custom code)
# ═══════════════════════════════════════════════════════════════════════════════
def partial_loglik(beta, X, t, e):
    """Mean partial log-likelihood (Breslow), scaled by n_events."""
    order   = np.argsort(t)
    X_s     = X[order]; t_s = t[order]; e_s = e[order]
    eta_s   = X_s @ beta
    eta_s  -= eta_s.max()
    exp_s   = np.exp(eta_s)
    cum_exp = np.cumsum(exp_s[::-1])[::-1]
    n_ev    = e_s.sum()
    ll      = 0.0
    for i in range(len(t_s)):
        if e_s[i]:
            ll += eta_s[i] - np.log(cum_exp[i] + 1e-300)
    return ll / (n_ev + 1e-300)

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  GRADIENT AT β=0  →  λ_max
#     Identical formula to custom code so λ grids are directly comparable.
# ═══════════════════════════════════════════════════════════════════════════════
def _grad_at_zero(X, t, e):
    """Gradient of negative partial log-likelihood at beta=0 (1/n_events)."""
    order   = np.argsort(t)
    X_s     = X[order]
    e_s     = e[order].astype(float)
    n_      = X_s.shape[0]
    # exp(0) = 1 everywhere
    cum_exp = np.cumsum(np.ones(n_)[::-1])[::-1]
    cum_Xw  = np.cumsum(X_s[::-1], axis=0)[::-1]
    n_ev    = e_s.sum()
    grad    = np.zeros(X.shape[1])
    for i in range(n_):
        if e_s[i]:
            grad += X_s[i] - cum_Xw[i] / (cum_exp[i] + 1e-300)
    return -grad / (n_ev + 1e-300)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  λ GRID + SOLUTION PATH
# ═══════════════════════════════════════════════════════════════════════════════
N_LAM     = 80
LAM_RATIO = 0.01

g0      = _grad_at_zero(X_std, t_all, e_float)
lam_max = float(np.max(np.abs(g0)))
lam_min = lam_max * LAM_RATIO
lam_grid = np.exp(np.linspace(np.log(lam_max), np.log(lam_min), N_LAM))

print(f"\nlambda grid: {lam_max:.4f} → {lam_min:.4f}  ({N_LAM} values)")
print("Fitting LASSO path  ...", end="", flush=True)

# ── Fit path one λ at a time ──────────────────────────────────────────────────
path_std = np.zeros((p, N_LAM))

for j, lam in enumerate(lam_grid):
    m = CoxnetSurvivalAnalysis(
        l1_ratio=1.0,
        alphas=[float(lam)],
        normalize=False,
        fit_baseline_model=False,
        max_iter=100_000,
        tol=1e-8,
    )
    m.fit(X_std, y_surv)
    path_std[:, j] = m.coef_.ravel()
    if j % 16 == 0:
        print(".", end="", flush=True)

print(" done")
path_orig = path_std / X_sd_full[:, None]   # back-transform to original scale

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  5-FOLD CROSS-VALIDATION  (held-out partial log-likelihood)
# ═══════════════════════════════════════════════════════════════════════════════
K         = 5
cv_scores = np.zeros((K, N_LAM))

print(f"\n{K}-fold CV  ...", end="", flush=True)

kf = KFold(n_splits=K, shuffle=True, random_state=42)

for k, (train_idx, val_idx) in enumerate(kf.split(X_std)):
    X_tr = X_std[train_idx];  y_tr = y_surv[train_idx]
    X_va = X_std[val_idx];    t_va = t_all[val_idx]
    e_va = e_float[val_idx]

    for j, lam in enumerate(lam_grid):
        m = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alphas=[float(lam)],
            normalize=False,
            fit_baseline_model=False,
            max_iter=100_000,
            tol=1e-7,
        )
        m.fit(X_tr, y_tr)
        beta_j = m.coef_.ravel()
        cv_scores[k, j] = partial_loglik(beta_j, X_va, t_va, e_va)

    print(f" fold{k+1}", end="", flush=True)

print(" done")

cv_mean  = cv_scores.mean(axis=0)
cv_se    = cv_scores.std(axis=0) / np.sqrt(K)
best_j   = np.argmax(cv_mean)
lam_best = lam_grid[best_j]
threshold = cv_mean[best_j] - cv_se[best_j]
one_se_j  = np.where(cv_mean >= threshold)[0][0]
lam_1se   = lam_grid[one_se_j]

print(f"\nlambda_min = {lam_best:.5f}  (index {best_j})")
print(f"lambda_1SE = {lam_1se:.5f}  (index {one_se_j})")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  REFIT ON FULL DATA at selected λ values
# ═══════════════════════════════════════════════════════════════════════════════
def refit_at_alpha(alpha):
    m = CoxnetSurvivalAnalysis(
        l1_ratio=1.0,
        alphas=[float(alpha)],
        normalize=False,
        fit_baseline_model=False,
        max_iter=200_000,
        tol=1e-10,
    )
    m.fit(X_std, y_surv)
    return m.coef_.ravel()

beta_min_std = refit_at_alpha(lam_best)
beta_1se_std = refit_at_alpha(lam_1se)
beta_min     = beta_min_std / X_sd_full
beta_1se     = beta_1se_std / X_sd_full
n_nonzero_min = int(np.sum(np.abs(beta_min_std) > 1e-6))
n_nonzero_1se = int(np.sum(np.abs(beta_1se_std) > 1e-6))

# ═══════════════════════════════════════════════════════════════════════════════
# 6b.  POST-FIT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  POST-FIT DIAGNOSTIC CHECKS  (lambda_min model)")
print("─" * 60)
COEF_THRESHOLD = 5.0
print(f"\n[Check 2] Runaway coefficients (|beta_std| > {COEF_THRESHOLD}):")
runaway = np.where(np.abs(beta_min_std) > COEF_THRESHOLD)[0]
if len(runaway):
    for k in runaway:
        print(f"  [WARNING] '{COVARIATE_NAMES[k]}':  "
              f"beta_std={beta_min_std[k]:+.3f},  beta_orig={beta_min[k]:+.3f}")
else:
    print("  No runaway coefficients detected.")
top_k = np.argmax(np.abs(beta_min_std))
print(f"\n  Largest |beta_std|: '{COVARIATE_NAMES[top_k]}' = "
      f"{beta_min_std[top_k]:+.4f}")
print("─" * 60 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  BOOTSTRAP 95% CIs  (λ_min model, fixed-λ, identical to custom code)
# ═══════════════════════════════════════════════════════════════════════════════
B         = 200
boot_coef = np.zeros((B, p))
print(f"Bootstrap CIs at lambda_min  (B={B})  ...", end="", flush=True)

for b in range(B):
    idx_b = np.random.choice(n, n, replace=True)
    X_b   = X_std[idx_b]
    y_b   = y_surv[idx_b]
    if y_b["event"].sum() < 5:
        boot_coef[b] = beta_min_std
        continue
    bm = CoxnetSurvivalAnalysis(
        l1_ratio=1.0,
        alphas=[float(lam_best)],
        normalize=False,
        fit_baseline_model=False,
        max_iter=100_000,
        tol=1e-7,
    )
    try:
        bm.fit(X_b, y_b)
        boot_coef[b] = bm.coef_.ravel()
    except Exception:
        boot_coef[b] = beta_min_std
    if b % 50 == 0:
        print(".", end="", flush=True)

print(" done")

boot_orig = boot_coef / X_sd_full[None, :]
ci_lo     = np.percentile(boot_orig, 2.5,  axis=0)
ci_hi     = np.percentile(boot_orig, 97.5, axis=0)
hr_min    = np.exp(beta_min)
hr_lo_min = np.exp(ci_lo)
hr_hi_min = np.exp(ci_hi)

# "unpenalized" = coefficients at lam_min (least regularised point on path)
beta_unpen = path_orig[:, -1]

important_idx = sorted(
    [k for k in range(p) if np.abs(beta_min_std[k]) > 1e-6],
    key=lambda k: abs(beta_min[k]), reverse=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 74)
print("  LASSO-PENALIZED COX PH MODEL (scikit-survival) — Summary")
print("═" * 74)
print(f"  lambda_min = {lam_best:.5f}  → {n_nonzero_min} non-zero  |  "
      f"lambda_1SE = {lam_1se:.5f}  → {n_nonzero_1se} non-zero")
print("\n  Reference categories:")
print("    region          → Africa (code 4)")
print("    incompatibility → Territory (code 1)")
print("    pa_type         → Full agreement (code 1)")
print("    frame           → Frame type 1 (code 1)")
print(f"\n── lambda_min model  (sorted by |beta|) ─────────────────────────────")
print(f"  {'Covariate':<47} {'beta':>8}  {'HR':>7}  95% Boot CI")
print("  " + "─" * 72)
for k in sorted(range(p), key=lambda k: abs(beta_min[k]), reverse=True):
    if abs(beta_min[k]) < 1e-6:
        continue
    b, hr      = beta_min[k], hr_min[k]
    lo, hi     = hr_lo_min[k], hr_hi_min[k]
    sig        = "  ***" if (lo > 1 or hi < 1) else ""
    print(f"  {LABELS[COVARIATE_NAMES[k]]:<47} {b:+8.4f}  {hr:7.3f}"
          f"  ({lo:.3f}-{hi:.3f}){sig}")

zeroed = [COVARIATE_NAMES[k] for k in range(p) if abs(beta_min[k]) < 1e-6]
print(f"\n  Zeroed-out ({len(zeroed)}): {', '.join(zeroed)}\n")

selected_interactions = [
    iname for iname in INTERACTION_NAMES
    if abs(beta_min[COVARIATE_NAMES.index(iname)]) > 1e-6
]
zeroed_interactions = [i for i in INTERACTION_NAMES if i not in selected_interactions]
if INTERACTIONS:
    print(f"  Selected interactions ({len(selected_interactions)}): "
          f"{', '.join(selected_interactions) or 'none'}")
    print(f"  Zeroed interactions  ({len(zeroed_interactions)}): "
          f"{', '.join(zeroed_interactions) or 'none'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  FIGURES  (identical layout/style to custom version)
# ═══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Georgia", "Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e0dcd5",
    "grid.linewidth":    0.55,
    "grid.linestyle":    "--",
    "figure.facecolor":  "#FAFAF7",
    "axes.facecolor":    "#FAFAF7",
    "axes.labelcolor":   "#2b2b2b",
    "xtick.color":       "#2b2b2b",
    "ytick.color":       "#2b2b2b",
    "text.color":        "#2b2b2b",
})
PALETTE = [
    "#E63946","#2A9D8F","#E9C46A","#457B9D","#F4A261","#6A4C93",
    "#264653","#E76F51","#06D6A0","#F77F00","#118AB2","#EF233C",
    "#8338EC","#3A86FF","#FB8500","#023047","#219EBC","#8ECAE6",
    "#606C38","#DDA15E",
]
imp_color = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(important_idx)}
n_show    = len(important_idx)
fig_h     = max(18, 4 + n_show * 0.55)
fig       = plt.figure(figsize=(20, fig_h), facecolor="#FAFAF7")
fig.suptitle(
    "Lasso-Penalized Cox PH (scikit-survival) — Peace Agreement Duration  "
    f"({p} predictors → {n_nonzero_min} selected at lambda_min)",
    fontsize=16, fontweight="bold", y=0.998, color="#1a1a1a",
)
fig.text(0.5, 0.983,
         f"lambda_min={lam_best:.4f} | lambda_1SE={lam_1se:.4f} | "
         f"Bootstrap 95% CIs (B={B}) | Event: agreement collapse",
         ha="center", fontsize=9.5, color="#555", style="italic")

gs      = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.61)
ax_path = fig.add_subplot(gs[0, 0])
ax_cv   = fig.add_subplot(gs[0, 1])
ax_fp   = fig.add_subplot(gs[1, 0])
ax_imp  = fig.add_subplot(gs[1, 1])

log_lam = np.log(lam_grid)

# ── A. Coefficient path ───────────────────────────────────────────────────────
# path_orig shape: (p, N_LAM)
for k in range(p):
    if k not in imp_color:
        ax_path.plot(log_lam, path_orig[k, :],
                     color="#d0d0d0", lw=0.7, alpha=0.45, zorder=1)
for k in important_idx:
    ax_path.plot(log_lam, path_orig[k, :],
                 color=imp_color[k], lw=2.2, alpha=0.95, zorder=3)

ax_path.axvline(np.log(lam_best), color="#333", lw=1.4, ls="--",
                label=f"lambda_min  ({n_nonzero_min} predictors)", zorder=4)
ax_path.axvline(np.log(lam_1se),  color="#999", lw=1.2, ls=":",
                label=f"lambda_1SE  ({n_nonzero_1se} predictors)", zorder=4)
ax_path.axhline(0, color="#aaa", lw=0.6, zorder=2)

n_active = (np.abs(path_std) > 1e-6).sum(axis=0)   # shape (N_LAM,)
tick_idx = np.linspace(0, N_LAM - 1, 9, dtype=int)
ax2 = ax_path.twiny()
ax2.set_xlim(ax_path.get_xlim())
ax2.set_xticks(log_lam[tick_idx])
ax2.set_xticklabels(n_active[tick_idx], fontsize=7.5)
ax2.set_xlabel("# non-zero coefficients", fontsize=8.5)
ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(False)

ax_path.invert_xaxis()
ax_path.set_xlabel("log(lambda)  <-  increasing regularisation", fontsize=10)
ax_path.set_ylabel("Coefficient (original scale)", fontsize=10)
ax_path.set_title("A.  Lasso Coefficient Path\n"
                  "(coloured = selected at lambda_min;  grey = zeroed out)",
                  fontsize=11.5, fontweight="bold", pad=10)

path_legend_handles = [
    plt.Line2D([0], [0], color=imp_color[k], lw=2.2,
               label=LABELS[COVARIATE_NAMES[k]].split(" [ref")[0])
    for k in important_idx
]
vline_handles = [
    plt.Line2D([0],[0], color="#333", lw=1.4, ls="--",
               label=f"lambda_min  ({n_nonzero_min} predictors)"),
    plt.Line2D([0],[0], color="#999", lw=1.2, ls=":",
               label=f"lambda_1SE  ({n_nonzero_1se} predictors)"),
]
ax_path.legend(
    handles=vline_handles + path_legend_handles,
    fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1),
    borderaxespad=0, framealpha=0.88, edgecolor="#ccc",
    title="Selected predictors", title_fontsize=8.5,
)

# ── B. Cross-validation curve ─────────────────────────────────────────────────
ax_cv.plot(log_lam, cv_mean, color="#457B9D", lw=2.2, label="Mean CV log-lik")
ax_cv.fill_between(log_lam, cv_mean - cv_se, cv_mean + cv_se,
                   alpha=0.18, color="#457B9D", label="+/- 1 SE")
ax_cv.axvline(np.log(lam_best), color="#E63946", lw=1.6, ls="--",
              label=f"lambda_min = {lam_best:.4f}  ({n_nonzero_min} vars)")
ax_cv.axvline(np.log(lam_1se),  color="#2A9D8F", lw=1.4, ls=":",
              label=f"lambda_1SE = {lam_1se:.4f}  ({n_nonzero_1se} vars)")
ymin, ymax = cv_mean.min(), cv_mean.max()
ax_cv.axvspan(log_lam[-1], np.log(lam_best), alpha=0.06, color="#E63946", zorder=0)
ax_cv.text((log_lam[-1] + np.log(lam_best)) / 2,
           ymin + (ymax - ymin) * 0.06,
           f"{n_nonzero_min} vars\nselected",
           ha="center", fontsize=8, color="#E63946", style="italic")
ax_cv.invert_xaxis()
ax_cv.set_xlabel("log(lambda)  <-  increasing regularisation", fontsize=10)
ax_cv.set_ylabel("Mean held-out partial log-lik", fontsize=10)
ax_cv.set_title(f"B.  {K}-Fold Cross-Validation Curve", fontsize=11.5,
                fontweight="bold", pad=10)
ax_cv.legend(fontsize=9, framealpha=0.88, edgecolor="#ccc")

# ── C. Forest plot ────────────────────────────────────────────────────────────
y_pos = np.arange(n_show)[::-1]
for i, (k, yi) in enumerate(zip(important_idx, y_pos)):
    c    = imp_color[k]
    b    = beta_min[k]
    lo   = ci_lo[k]; hi = ci_hi[k]
    excl = lo > 0 or hi < 0
    ax_fp.plot([lo, hi], [yi, yi], color=c, lw=2.4,
               solid_capstyle="round", zorder=3)
    ax_fp.scatter(b, yi, color=c, s=70, zorder=5,
                  marker="D" if excl else "o",
                  edgecolors="white", linewidths=0.7)
    ax_fp.text(hi + 0.02, yi,
               f"{b:+.3f} ({lo:+.3f}-{hi:+.3f})",
               va="center", fontsize=7.0, color="#333")
ax_fp.axvline(0.0, color="#888", lw=1.3, ls="--", zorder=1)
ax_fp.set_yticks(y_pos)
ax_fp.set_yticklabels([LABELS[COVARIATE_NAMES[k]] for k in important_idx],
                      fontsize=8.2)
for ytick, k in zip(ax_fp.get_yticklabels(), important_idx):
    ytick.set_color(imp_color[k])
ax_fp.set_xlabel("Coefficient beta  (95% bootstrap CI)", fontsize=10)
ax_fp.set_title(
    f"C.  Forest Plot — {n_nonzero_min} Selected Predictors at lambda_min\n"
    f"(diamond = CI excludes 0;  sorted by |beta|)",
    fontsize=11.5, fontweight="bold", pad=10,
)

# ── D. Variable importance ────────────────────────────────────────────────────
y_b   = np.arange(n_show)
bw    = 0.38
short = [LABELS[COVARIATE_NAMES[k]].split(" [ref")[0] for k in important_idx]
for i, k in enumerate(important_idx):
    c     = imp_color[k]
    sign  = np.sign(beta_min[k])
    b_las = abs(beta_min[k])
    b_unp = abs(beta_unpen[k])
    ax_imp.barh(y_b[i] + bw / 2, b_las, bw, color=c, alpha=0.90,
                label="Lasso (lambda_min)" if i == 0 else "")
    ax_imp.barh(y_b[i] - bw / 2, b_unp, bw, color=c, alpha=0.30,
                label="Unpenalized (lam_min path)" if i == 0 else "")
    d_color = "#C0392B" if sign > 0 else "#1A7A4A"
    ax_imp.text(max(b_las, b_unp) + 0.008, y_b[i],
                " ▲" if sign > 0 else " ▼",
                va="center", fontsize=9, color=d_color, fontweight="bold",
                fontfamily="DejaVu Sans")
ax_imp.set_yticks(y_b)
ax_imp.set_yticklabels(short, fontsize=8.2)
for ytick, k in zip(ax_imp.get_yticklabels(), important_idx):
    ytick.set_color(imp_color[k])
ax_imp.set_xlabel("Coefficient Magnitude", fontsize=10)
ax_imp.set_title(
    "D.  Variable Importance\n"
    "(▲ = increases hazard  /  ▼ = reduces hazard)",
    fontsize=11.5, fontweight="bold", pad=10, fontfamily="DejaVu Sans"
)
legend_handles = [
    Patch(color="#777", alpha=0.90, label="LASSO"),
    Patch(color="#777", alpha=0.30, label="Unpenalized"),
]
ax_imp.legend(handles=legend_handles, fontsize=8.5,
              framealpha=0.88, edgecolor="#ccc", loc="upper right")

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.expanduser("~/thesis/surv_modeling")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "cox_lasso_results.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#FAFAF7")
print(f"\nFigure saved → {out}")
plt.close()