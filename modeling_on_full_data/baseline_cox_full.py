"""
Cox Proportional Hazards Model — Peace Agreement Duration
==========================================================
Outcome  : span_days (time to agreement collapse)
Event    : ended == 1  (agreement collapsed)
Censored : ended is NaN or 0

Design choices
--------------
1. Predictors standardized (mean 0, sd 1) before fitting.
   Coefficients back-transformed to original scale for reporting.

2. Interaction terms built as products of STANDARDIZED variables.
   Because each variable is already mean-centred after standardisation,
   the product term is orthogonal to both main effects — this removes
   the lower-order correlations that would otherwise confound the
   interaction estimate. Main effects are retained in the model.

3. Proportional-hazards assumption tested via the Grambsch-Therneau
   method: Pearson correlation of the scaled Schoenfeld residual with
   log(t). This is the direct equivalent of R's cox.zph().

4. Variables that fail the PH test (p < PH_ALPHA) are re-entered in an
   extended Cox model as  x + x*log(t), implementing
       beta(t) = beta_1 + beta_2 * log(t)
   Each subject's own observed time is used for log(t), which is the
   standard Grambsch-Therneau approach and carries no data leakage.

5. Cluster-robust (Lin-Wei sandwich) SEs clustered by conflict_id
   for both the main and extended models.

Outputs
-------
  • Console: univariate table, main multivariable table, PH test,
             extended model table
  • baseline_results.png — five-panel figure:
      A  Forest plot — univariate HRs
      B  Forest plot — multivariable HRs (cluster-robust CIs)
      C  Breslow baseline cumulative hazard
      D  Schoenfeld residual plots (PH check)
      E  Forest plot — extended model (time-varying terms)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from numpy.linalg import lstsq
from scipy.stats import norm
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

warnings.filterwarnings("ignore")

# ── tunable threshold for PH violation ───────────────────────────────────────
PH_ALPHA = 0.05

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA PREP
# ═══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv("pad_merged.csv")
df = df.rename(columns={"ref ": "ref", "return ": "return"})

df["censor_point"] = pd.to_datetime(df["censor_point"])
df["pa_date"]      = pd.to_datetime(df["pa_date"])

censored_mask = df["ended"].isna() | (df["ended"] != 1)
df.loc[censored_mask, "span_days"] = (
    df.loc[censored_mask, "censor_point"]
    - df.loc[censored_mask, "pa_date"]
).dt.days

df["ended"] = df["ended"].fillna(0)

n_before = len(df)
df = df.dropna(subset=["span_days"]).copy()
n_after = len(df)
if n_before != n_after:
    print(f"  [INFO] Dropped {n_before - n_after} rows with NaN in "
          f"span_days  ({n_after} rows remain)")

df["event"] = (df["ended"] == 1).astype(bool)
df["t"]     = df["span_days"].astype(float)

print(f"Rows: {len(df)}  |  Events: {int(df['event'].sum())}  "
      f"|  Censored: {int((~df['event']).sum())}  "
      f"|  NaN span_days remaining: {df['span_days'].isna().sum()}")


# ── External variable prep (merged dataset) ───────────────────────────────────
# Log-transform skewed continuous variables (log1p guards against 0)
# NOTE: gdp_per_capita and gdp_growth in pad_merged.csv are already
#       constructed as one-year lags (t-1) relative to the signing year.
#       No additional lagging is applied here.
df["log_gdp_pc"]   = np.log1p(df["gdp_per_capita"])
df["log_milper"]   = np.log1p(df["milper"])
df["log_bd_total"] = np.log1p(df["bd_best_total"])
df["log_v2regdur"] = np.log1p(df["v2regdur"])

# Prior outcome: flag unknown cases (31 rows), fill binary vars with 0
df["unknown_outcome"]    = df["rebel_victory"].isna().astype(float)
df["rebel_victory"]      = df["rebel_victory"].fillna(0).astype(float)
df["government_victory"] = df["government_victory"].fillna(0).astype(float)
df["regime_change"]      = df["regime_change"].fillna(0).astype(float)
df["neighbor_at_war"]    = df["neighbor_at_war"].fillna(0).astype(float)

# Median-impute remaining NaN in external continuous variables
_ext_impute = [
    "lmtnest", "log_gdp_pc", "gdp_growth", "log_milper",
    "v2x_polyarchy", "log_v2regdur", "v2x_freexp_altinf",
    "v2x_liberal", "v2x_partip", "log_bd_total",
]
for _col in _ext_impute:
    _n = df[_col].isna().sum()
    if _n > 0:
        _med = df[_col].median()
        df[_col] = df[_col].fillna(_med)
        print(f"  [IMPUTE] '{_col}': {_n} NaN → median ({_med:.4f})")

print(f"  External variables prepared. unknown_outcome=1 for "
      f"{int(df['unknown_outcome'].sum())} rows.")

# ── covariate lists ───────────────────────────────────────────────────────────
BINARY = [
    "cease", "intarmy", "ddr", "withd", "pp", "intgov", "intciv",
    "elections", "interim", "natalks", "aut", "fed", "ind", "ref",
    "shaloc", "regdev", "cul", "demarcation", "locgov", "amn", "pris",
    "recon", "return", "reaffirm", "outlin", "pko", "gender", "co_impl",
]
ORDINAL = ["termdur"]

# External variables from merged dataset (literature review factors)
EXTERNAL_BIN = [
    "neighbor_at_war",    # Hegre & Sambanis 2006: war-prone regional neighborhood
    "regime_change",      # Hegre & Sambanis 2006: political volatility
    "rebel_victory",      # Quinn et al. 2007: rebel victory in previous conflict
    "government_victory", # Quinn et al. 2007: government victory in previous conflict
    "unknown_outcome",    # Flag: prior conflict outcome not in UCDP (31 cases)
]
EXTERNAL_CONT = [
    "lmtnest",            # Hegre & Sambanis 2006: % mountainous terrain (log)
    "log_gdp_pc",         # Control: log GDP per capita, lagged one year (t-1)
    "gdp_growth",         # Hegre & Sambanis 2006 + Walter 2015: economic growth (t-1)
    "log_milper",         # Hegre & Sambanis 2006: state military size (log)
    "v2x_polyarchy",      # V-Dem electoral democracy index (0-1)
    "log_v2regdur",       # Regime duration in days (log) — regime stability
    "v2x_freexp_altinf",  # Walter 2015: media freedom (0-1)
    "v2x_liberal",        # Walter 2015: liberal political institutions (0-1)
    "v2x_partip",         # Walter 2015: political participation (0-1)
    "log_bd_total",       # Quinn et al. 2007: prior conflict death toll (log)
]


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

BASE_NAMES = BINARY + EXTERNAL_BIN + ORDINAL + EXTERNAL_CONT + DUMMIES

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
INTERACTION_NAMES = [f"{a}::{b}" for (a, b) in INTERACTIONS]

for a, b in INTERACTIONS:
    assert a in BASE_NAMES, f"'{a}' not in BASE_NAMES"
    assert b in BASE_NAMES, f"'{b}' not in BASE_NAMES"

ALL_NAMES = BASE_NAMES + INTERACTION_NAMES   # 52 columns

# ── labels ────────────────────────────────────────────────────────────────────
LABELS = {
    "region_europe":       "Region: Europe [ref: Africa]",
    "region_middleeast":   "Region: Middle East [ref: Africa]",
    "region_asia":         "Region: Asia [ref: Africa]",
    "region_americas":     "Region: Americas [ref: Africa]",
    "cease":               "Ceasefire provision",
    "intarmy":             "Integration of armed forces",
    "ddr":                 "DDR provision",
    "withd":               "Withdrawal provision",
    "pp":                  "Power-sharing (political)",
    "intgov":              "Integration into government",
    "intciv":              "Civilian power-sharing",
    "elections":           "Elections provision",
    "interim":             "Rebels into interim government",
    "natalks":             "National talks",
    "aut":                 "Autonomy provision",
    "fed":                 "Federalism provision",
    "ind":                 "Independence provision",
    "ref":                 "Referendum provision",
    "shaloc":              "Local power-sharing (shaloc)",
    "regdev":              "Regional development",
    "cul":                 "Cultural provisions",
    "demarcation":         "Border demarcation",
    "locgov":              "Local governance",
    "amn":                 "Amnesty provision",
    "pris":                "Prisoner release",
    "recon":               "Reconciliation",
    "return":              "Return of refugees",
    "reaffirm":            "Reaffirmation of prev. agreement",
    "outlin":              "Outline agreement",
    "pko":                 "Peacekeeping operation",
    "gender":              "Gender provision",
    "co_impl":             "Oversight committee",
    "termdur":             "Years since last conflict",
    "incomp_gov":          "Incompatibility: Government [ref: Territory]",
    "incomp_both":         "Incompatibility: Both [ref: Territory]",
    "patype_partial":      "Agreement type: Partial [ref: Full]",
    "patype_process":      "Agreement type: Peace Process [ref: Full]",
    "frame_frame2":        "Frame type 2 [ref: Frame 1]",
    "frame_frame3":        "Frame type 3 [ref: Frame 1]",
    # ── External variables (merged dataset) ───────────────────────────────────
    "neighbor_at_war":    "Neighbor at war (any contiguous)",
    "regime_change":      "Regime change in signing year",
    "rebel_victory":      "Rebel victory (prior conflict)",
    "government_victory": "Government victory (prior conflict)",
    "unknown_outcome":    "Prior conflict outcome unknown",
    "lmtnest":            "Mountainous terrain (log % area)",
    "log_gdp_pc":         "GDP per capita, log (t-1)",
    "gdp_growth":         "GDP growth rate (t-1)",
    "log_milper":         "Military personnel, log (signing yr)",  # NMC: signing year
    "v2x_polyarchy":      "Electoral democracy index (V-Dem)",
    "log_v2regdur":       "Regime duration, log days (V-Dem)",
    "v2x_freexp_altinf":  "Media freedom index (V-Dem)",
    "v2x_liberal":        "Liberal institutions index (V-Dem)",
    "v2x_partip":         "Political participation index (V-Dem)",
    "log_bd_total":       "Prior conflict death toll, log",
    "cease::pko":          "Ceasefire  ×  Peacekeeping operation",
    "ddr::intarmy":        "DDR provision  ×  Integration of armed forces",
    "ddr::intgov":         "DDR provision  ×  Interim government",
    "pp::elections":       "Power-sharing (political)  ×  Elections provision",
    "amn::co_impl":        "Amnesty provision  ×  Co-implementation body",
    "shaloc::return":      "Local power-sharing  ×  Return / resettlement",
    "aut::shaloc":         "Autonomy provision  ×  Local power-sharing",
    "withd::pko":          "Withdrawal provision  ×  Peacekeeping operation",
    "reaffirm::co_impl":   "Reaffirm. prev. agreement  ×  Co-implementation body",
    "demarcation::locgov": "Border demarcation  ×  Local governance",
    "pris::amn":           "Prisoner release  ×  Amnesty provision",
    "amn::recon":          "Amnesty provision  ×  Reconciliation",
    "intgov::natalks":     "Interim government  ×  National dialogue / talks",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  BUILD DESIGN MATRIX
#     Step 1 — extract raw base columns
#     Step 2 — standardise (mean 0, sd 1)  →  variables are now centred
#     Step 3 — build interaction terms as products of STANDARDIZED columns
#              design matrix {p} columns  (residualized interactions)
#     Step 4 — stack into X_std  (N × 52)
#     Store X_mean, X_sd for back-transformation of coefficients later.
# ═══════════════════════════════════════════════════════════════════════════════

X_base_raw = df[BASE_NAMES].values.astype(float)

X_mean = X_base_raw.mean(axis=0)          # shape (39,)
X_sd   = X_base_raw.std(axis=0)
X_sd[X_sd == 0] = 1.0                     # guard against zero-variance columns
X_base_std = (X_base_raw - X_mean) / X_sd  # shape (N, 39)

# ── Build interactions via residualization ────────────────────────────────────
# Regress (x_a_std * x_b_std) on x_a_std and x_b_std and keep the residual.
# Guarantees orthogonality to both main effects regardless of variable type.

n = X_base_std.shape[0]
ia_resid = []
for a, b in INTERACTIONS:
    col_a    = X_base_std[:, BASE_NAMES.index(a)]
    col_b    = X_base_std[:, BASE_NAMES.index(b)]
    product  = col_a * col_b
    Z        = np.column_stack([np.ones(n), col_a, col_b])
    coefs, _, _, _ = lstsq(Z, product, rcond=None)
    ia_resid.append(product - Z @ coefs)

X_ia     = np.column_stack(ia_resid)
ia_mean  = X_ia.mean(axis=0)
ia_sd    = X_ia.std(axis=0)
ia_sd[ia_sd == 0] = 1.0
X_ia_std = (X_ia - ia_mean) / ia_sd

X_std = np.hstack([X_base_std, X_ia_std])  # shape (N, 52)

# Concatenate means and sds for back-transformation
ALL_MEAN = np.concatenate([X_mean, ia_mean])  # shape (52,)
ALL_SD   = np.concatenate([X_sd,   ia_sd  ])

n, p = X_std.shape
print(f"  + {len(INTERACTIONS)} interactions (centred × centred) → "
      f"design matrix {p} columns  (standardized)")
print(f"\n  Covariates: {len(BINARY)} PAD binary + {len(EXTERNAL_BIN)} external binary, "
      f"{len(ORDINAL)} ordinal + {len(EXTERNAL_CONT)} external continuous, "
      f"{len(DUMMIES)} dummies, {len(INTERACTIONS)} interactions\n")

# ── outcome arrays ────────────────────────────────────────────────────────────
t_all       = df["t"].values.astype(float)
event_bool  = df["event"].values
event_float = event_bool.astype(float)
y_surv      = Surv.from_arrays(event=event_bool, time=t_all)
cluster_ids = df["conflict_id"].values

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CORE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _neg_partial_loglik(beta, X, t, event):
    """Breslow neg partial log-likelihood."""
    eta     = X @ beta
    eta_max = eta.max()
    exp_eta = np.exp(eta - eta_max)
    order   = np.argsort(t)
    exp_s   = exp_eta[order]; e_s = event[order]
    cum_exp = np.cumsum(exp_s[::-1])[::-1]
    lpl = 0.0
    for i in range(len(e_s)):
        if e_s[i]:
            lpl += (eta[order[i]] - eta_max) - np.log(cum_exp[i] + 1e-300)
    return -lpl


def _numerical_hessian(beta, X, t, event, eps=1e-5):
    """Numerical Hessian of neg partial log-likelihood at beta."""
    q = len(beta)
    H = np.zeros((q, q))
    for i in range(q):
        for j in range(i, q):
            ei, ej = np.zeros(q), np.zeros(q)
            ei[i] = eps; ej[j] = eps
            H[i, j] = H[j, i] = (
                _neg_partial_loglik(beta + ei + ej, X, t, event)
              - _neg_partial_loglik(beta + ei - ej, X, t, event)
              - _neg_partial_loglik(beta - ei + ej, X, t, event)
              + _neg_partial_loglik(beta - ei - ej, X, t, event)
            ) / (4 * eps ** 2)
    return H


def _ses_from_hessian(beta, X, t, event):
    H = _numerical_hessian(beta, X, t, event)
    try:
        return np.sqrt(np.maximum(np.diag(np.linalg.inv(H)), 0))
    except np.linalg.LinAlgError:
        return np.full(len(beta), np.nan)


def _score_contributions(beta, X, t, event):
    """Per-observation score vectors s_i."""
    n_, q   = X.shape
    exp_eta = np.exp(X @ beta)
    order   = np.argsort(t)
    X_s     = X[order]; e_s = event[order]
    exp_s   = exp_eta[order]
    cum_exp = np.cumsum(exp_s[::-1])[::-1]
    cum_Xw  = np.cumsum((X_s * exp_s[:, None])[::-1], axis=0)[::-1]
    ss = np.zeros((n_, q))
    for i in range(n_):
        if e_s[i]:
            ss[i] = X_s[i] - cum_Xw[i] / (cum_exp[i] + 1e-300)
    scores = np.zeros((n_, q))
    scores[order] = ss
    return scores


def _schoenfeld(beta, X, t, event):
    """
    Scaled Schoenfeld residuals.
    Returns (event_times array, residual matrix (n_events × q)).
    """
    exp_eta = np.exp(X @ beta)
    order   = np.argsort(t)
    t_s     = t[order]; e_s = event[order]
    exp_s   = exp_eta[order]; X_s = X[order]
    cum_exp = np.cumsum(exp_s[::-1])[::-1]
    cum_Xw  = np.cumsum((X_s * exp_s[:, None])[::-1], axis=0)[::-1]
    etimes, resids = [], []
    for i in range(len(t_s)):
        if e_s[i]:
            etimes.append(t_s[i])
            resids.append(X_s[i] - cum_Xw[i] / (cum_exp[i] + 1e-300))
    return np.array(etimes), np.array(resids)


def _ph_test(event_times, resids, names):
    """
    Grambsch-Therneau PH test: Pearson rho of Schoenfeld residual with log(t).
    Equivalent to R's cox.zph(transform='log').
    Returns {name: (rho, p)}.
    """
    from scipy.stats import pearsonr
    log_t   = np.log(event_times + 1)
    results = {}
    for k, name in enumerate(names):
        r    = resids[:, k]
        mask = np.isfinite(r) & np.isfinite(log_t)
        r_c, t_c = r[mask], log_t[mask]
        if len(r_c) < 3 or np.std(r_c) < 1e-10:
            results[name] = (np.nan, np.nan)
        else:
            rho, pv = pearsonr(t_c, r_c)
            results[name] = (rho, pv)
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FIT WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fit_sksurv(X, y, t, event):
    """
    Fit Cox PH via CoxPHSurvivalAnalysis.
    Returns coefficients on the STANDARDIZED scale together with naive SEs.
    """
    m = CoxPHSurvivalAnalysis(ties="breslow", n_iter=500, tol=1e-9)
    m.fit(X, y)
    beta   = m.coef_
    loglik = -_neg_partial_loglik(beta, X, t, event)
    se     = _ses_from_hessian(beta, X, t, event)
    z      = beta / (se + 1e-300)
    pv     = 2 * norm.sf(np.abs(z))
    return dict(model=m, beta=beta, se=se, z=z, p=pv,
                hr=np.exp(beta), hr_lo=np.exp(beta - 1.96*se),
                hr_hi=np.exp(beta + 1.96*se),
                loglik=loglik, n=len(t),
                n_events=int(event.sum()), converged=True)


def fit_clustered(X, y, t, event, cluster_ids):
    """
    Cox PH with Lin-Wei sandwich SEs (clustered by conflict_id).
    Point estimates from sksurv; variance from H^{-1} B H^{-1}.
    """
    base  = fit_sksurv(X, y, t, event)
    beta  = base["beta"]
    cids  = np.asarray(cluster_ids)
    q     = len(beta)

    scores   = _score_contributions(beta, X, t, event)
    clusters = np.unique(cids)
    B = np.zeros((q, q))
    for c in clusters:
        sc = scores[cids == c].sum(axis=0)
        B += np.outer(sc, sc)

    H = _numerical_hessian(beta, X, t, event)
    try:
        Hi    = np.linalg.inv(H)
        se_cl = np.sqrt(np.maximum(np.diag(Hi @ B @ Hi), 0))
    except np.linalg.LinAlgError:
        print("  [WARNING] Hessian singular — clustered SEs unavailable.")
        se_cl = np.full(q, np.nan)

    z_cl = beta / (se_cl + 1e-300)
    p_cl = 2 * norm.sf(np.abs(z_cl))
    return dict(
        model      = base["model"],
        beta       = beta,
        se         = se_cl,
        se_naive   = base["se"],
        z          = z_cl,
        p          = p_cl,
        hr         = base["hr"],
        hr_lo      = np.exp(beta - 1.96 * se_cl),
        hr_hi      = np.exp(beta + 1.96 * se_cl),
        loglik     = base["loglik"],
        n          = len(t),
        n_events   = int(event.sum()),
        n_clusters = len(clusters),
        converged  = base["converged"],
    )


def back_transform(res, sd_vec):
    """
    Convert standardized-scale beta/SE/HR to original scale.
    beta_orig  = beta_std / sd
    se_orig    = se_std / sd
    HR and CIs recomputed from back-transformed beta/se.
    """
    b    = res["beta"] / sd_vec
    se   = res["se"]   / sd_vec
    se_n = res.get("se_naive", res["se"]) / sd_vec
    z    = b / (se + 1e-300)
    pv   = 2 * norm.sf(np.abs(z))
    return dict(
        beta     = b,
        se       = se,
        se_naive = se_n,
        z        = z,
        p        = pv,
        hr       = np.exp(b),
        hr_lo    = np.exp(b - 1.96 * se),
        hr_hi    = np.exp(b + 1.96 * se),
        loglik   = res["loglik"],
        n        = res["n"],
        n_events = res["n_events"],
        n_clusters = res.get("n_clusters", None),
        converged  = res["converged"],
    )


def breslow_baseline(model, X):
    X_ref  = np.zeros((1, X.shape[1]))
    chf_fn = model.predict_cumulative_hazard_function(X_ref)[0]
    return chf_fn.x, chf_fn.y

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FIT MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 72)
print("  COX PH MODEL  (scikit-survival, standardized predictors)")
print("═" * 72)
print(f"  N = {n}   Events = {int(event_float.sum())}   "
      f"Censored = {int((event_float==0).sum())}")
print("\n  Reference categories:")
print("    region → Africa  |  incompatibility → Territory  |  "
      "pa_type → Full  |  frame → Frame 1")

# ── A. Univariate models ──────────────────────────────────────────────────────
print("\n── UNIVARIATE MODELS ────────────────────────────────────────────────")
print(f"  {'Covariate':<50} {'HR':>7} {'95% CI':>18}  {'p':>8}")
print("  " + "─" * 84)

uni_results = {}
for j, col in enumerate(ALL_NAMES):
    X_u   = X_std[:, j:j+1]
    y_u   = Surv.from_arrays(event=event_bool, time=t_all)
    res_s = fit_sksurv(X_u, y_u, t_all, event_float)
    # back-transform with scalar sd
    res_o = back_transform(res_s, np.array([ALL_SD[j]]))
    uni_results[col] = res_o
    hr, lo, hi, pv = res_o["hr"][0], res_o["hr_lo"][0], res_o["hr_hi"][0], res_o["p"][0]
    stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
    print(f"  {LABELS[col]:<50} {hr:7.3f}  ({lo:.3f} - {hi:.3f})  {pv:8.4f} {stars}")

# ── B. Multivariable model ────────────────────────────────────────────────────
print("\nFitting multivariable model  ...", end="", flush=True)
multi_std = fit_clustered(X_std, y_surv, t_all, event_float, cluster_ids)
multi     = back_transform(multi_std, ALL_SD)
print(" done")

print("\n── MULTIVARIABLE MODEL (cluster-robust SEs, clustered by conflict_id) ──")
print(f"  Log-likelihood : {multi['loglik']:.3f}   "
      f"Converged: {multi['converged']}")
print(f"  N clusters     : {multi['n_clusters']}")
print(f"\n  {'Covariate':<50} {'HR':>7} {'Robust 95% CI':>18}  {'p':>8}  "
      f"SE-naive  SE-robust")
print("  " + "─" * 96)
for k, col in enumerate(ALL_NAMES):
    hr, lo, hi, pv = (multi["hr"][k], multi["hr_lo"][k],
                      multi["hr_hi"][k], multi["p"][k])
    stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
    print(f"  {LABELS[col]:<50} {hr:7.3f}  ({lo:.3f} - {hi:.3f})  "
          f"{pv:8.4f} {stars}   {multi['se_naive'][k]:.4f}    {multi['se'][k]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  GRAMBSCH-THERNEAU PH TEST
#     Uses the STANDARDIZED-scale betas (Schoenfeld residuals are scale-free).
# ═══════════════════════════════════════════════════════════════════════════════

e_times, resids = _schoenfeld(multi_std["beta"], X_std, t_all, event_float)
ph_tests        = _ph_test(e_times, resids, ALL_NAMES)

violators = [col for col in ALL_NAMES
             if not np.isnan(ph_tests[col][1]) and ph_tests[col][1] < PH_ALPHA]

print(f"\n── PH ASSUMPTION  (Grambsch-Therneau, transform = log(t)) ──────────")
print(f"  {'Covariate':<50} {'rho':>7}  {'p':>8}  Concern?")
print("  " + "─" * 74)
for col in ALL_NAMES:
    rho, pv  = ph_tests[col]
    flag     = " !" if col in violators else ""
    p_str    = f"{pv:.4f}" if not np.isnan(pv) else "  n/a"
    r_str    = f"{rho:.3f}" if not np.isnan(rho) else "  n/a"
    print(f"  {LABELS[col]:<50} {r_str:>7}  {p_str:>8}{flag}")

print(f"\n  PH violators (p < {PH_ALPHA}):  "
      f"{', '.join(violators) if violators else 'none'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  EXTENDED MODEL  —  x + x*log(t)  for PH violators
#
#     Correct approach: person-period (start-stop / counting process) expansion.
#
#     For each failure time t_k in the dataset, every observation still at risk
#     contributes one row.  The time-varying interaction x_j * log(t_k) is
#     evaluated at the *current failure time* t_k, not the subject's own
#     eventual exit time.  This avoids leakage: at the moment the likelihood
#     is evaluated for event t_k we only use information available then.
#
#     Dataset structure:
#         t_start  — previous failure time (0 for the first interval)
#         t_stop   — current failure time t_k
#         event    — 1 only for the observation whose event occurs at t_k
#         x_j * log(t_k) — interaction evaluated at t_k for everyone at risk
#
#     The extended model estimates:
#         beta(t) = beta_1  +  beta_2 * log(t)
#     where beta_1 is the log-HR when log(t) = 0 (i.e. t = 1 day) and beta_2
#     captures how the log-HR changes across the log-time scale.
# ═══════════════════════════════════════════════════════════════════════════════

EXT_NAMES  = []    # names of the added x*log(t) columns
EXT_LABELS = {}

if violators:
    # ── All unique failure times in the dataset ───────────────────────────────
    event_times_unique = np.sort(np.unique(t_all[event_bool]))

    # ── Build the violator-column indices once ────────────────────────────────
    viol_idx = [ALL_NAMES.index(col) for col in violators]

    # ── Person-period expansion ───────────────────────────────────────────────
    # For each observation i and each failure time t_k <= t_i, emit one row:
    #   - original standardised covariates X_std[i]
    #   - one extra column per violator: X_std[i, j] * log(t_k + 1)
    #   - (t_start, t_stop, is_event, cluster_id)
    pp_rows = []
    for i in range(len(t_all)):
        ti  = t_all[i]
        ei  = event_bool[i]
        cid = cluster_ids[i]
        xi  = X_std[i]

        # Failure times at which this observation is still in the risk set
        risk_event_times = event_times_unique[event_times_unique <= ti]

        for k_idx, tk in enumerate(risk_event_times):
            t_start  = risk_event_times[k_idx - 1] if k_idx > 0 else 0.0
            t_stop   = tk
            is_event = bool(ei) and (ti == tk)
            log_tk   = np.log(tk + 1)          # +1 guards against tk = 0

            # Time-varying interaction terms evaluated at t_k
            tv_vals = xi[viol_idx] * log_tk    # shape (n_violators,)

            pp_rows.append(
                np.concatenate([xi, tv_vals,
                                [t_start, t_stop, float(is_event), float(cid)]])
            )

    pp_col_names = (ALL_NAMES
                    + [f"{col}:log(t)" for col in violators]
                    + ["t_start", "t_stop", "is_event", "conflict_id_pp"])
    df_pp = pd.DataFrame(pp_rows, columns=pp_col_names)

    print(f"\n  Person-period dataset: {len(df_pp):,} rows  "
          f"(from {len(t_all)} original observations, "
          f"{len(event_times_unique)} unique failure times)")

    # ── Interaction (tv) column names and labels ──────────────────────────────
    for col in violators:
        tname = f"{col}:log(t)"
        EXT_NAMES.append(tname)
        EXT_LABELS[tname] = f"{LABELS[col].split(' [ref')[0]}  ×  log(t)"

    EXT_ALL = ALL_NAMES + EXT_NAMES

    # ── Design matrix and outcome for the extended model ─────────────────────
    X_ext_pp   = df_pp[EXT_ALL].values.astype(float)
    t_start_pp = df_pp["t_start"].values.astype(float)
    t_stop_pp  = df_pp["t_stop"].values.astype(float)
    event_pp   = df_pp["is_event"].values.astype(bool)
    cluster_pp = df_pp["conflict_id_pp"].values.astype(int)

    # Standardise only the new tv columns (x_std * log(t_k) is not unit-free);
    # the base columns are already standardised from the main model.
    tv_arr    = X_ext_pp[:, len(ALL_NAMES):]
    tv_mean   = tv_arr.mean(axis=0)
    tv_sd     = tv_arr.std(axis=0);  tv_sd[tv_sd == 0] = 1.0
    X_ext_pp_std = np.hstack([X_ext_pp[:, :len(ALL_NAMES)],
                               (tv_arr - tv_mean) / tv_sd])
    EXT_SD = np.concatenate([ALL_SD, tv_sd])

    y_ext = Surv.from_arrays(event=event_pp, time=t_stop_pp)

    print(f"\nFitting extended model  "
          f"({len(violators)} time-varying term(s): "
          f"{', '.join(violators)})  ...", end="", flush=True)
    ext_std = fit_clustered(X_ext_pp_std, y_ext, t_stop_pp,
                            event_pp.astype(float), cluster_pp)
    ext     = back_transform(ext_std, EXT_SD)
    print(" done")

    FULL_LABELS = {**LABELS, **EXT_LABELS}

    print(f"\n── EXTENDED MODEL  (cluster-robust SEs) ────────────────────────────")
    print(f"  Log-likelihood : {ext['loglik']:.3f}")
    print(f"\n  {'Covariate':<55} {'HR':>7} {'Robust 95% CI':>18}  {'p':>8}")
    print("  " + "─" * 90)
    for k, col in enumerate(EXT_ALL):
        hr, lo, hi, pv = (ext["hr"][k], ext["hr_lo"][k],
                          ext["hr_hi"][k], ext["p"][k])
        stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        marker = "  [tv]" if col in EXT_NAMES else ""
        print(f"  {FULL_LABELS[col]:<55} {hr:7.3f}  ({lo:.3f} - {hi:.3f})  "
              f"{pv:8.4f} {stars}{marker}")
    print("\n  [tv] = time-varying coefficient term  (x × log(t))")
else:
    print("\n  No PH violations detected — extended model not needed.")
    ext        = None
    EXT_ALL    = ALL_NAMES
    FULL_LABELS = LABELS

print("\n  *** p<0.001  ** p<0.01  * p<0.05\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  FIGURE
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

PALETTE = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261", "#6A4C93"]

n_panels = 5 if ext is not None else 4
# Forest plots A and B scale with the number of predictors; give them more
# vertical room than the diagnostic panels
fig_h = max(28, 6 + len(ALL_NAMES) * 0.42)
fig = plt.figure(figsize=(20, fig_h), facecolor="#FAFAF7")
fig.suptitle(
    "Cox Proportional Hazards Model — Peace Agreement Duration",
    fontsize=17, fontweight="bold", y=0.995, color="#1a1a1a",
)
fig.text(
    0.5, 0.980,
    f"Predictors standardized (mean 0, sd 1)  |  "
    f"Interactions residualized against main effects  |  "
    f"Cluster-robust SEs (Lin-Wei) by conflict_id  |  "
    f"N clusters = {multi['n_clusters']}",
    ha="center", fontsize=8.5, color="#555", style="italic",
)

if ext is not None:
    gs = fig.add_gridspec(3, 2, height_ratios=[3.5, 2.0, 2.0],
                          hspace=0.44, wspace=0.38, top=0.96)
    ax_uni   = fig.add_subplot(gs[0, 0])
    ax_multi = fig.add_subplot(gs[0, 1])
    ax_bh    = fig.add_subplot(gs[1, 0])
    ax_sch   = fig.add_subplot(gs[1, 1])
    ax_ext   = fig.add_subplot(gs[2, :])
else:
    gs = fig.add_gridspec(2, 2, height_ratios=[3.5, 2.0],
                          hspace=0.44, wspace=0.38, top=0.96)
    ax_uni   = fig.add_subplot(gs[0, 0])
    ax_multi = fig.add_subplot(gs[0, 1])
    ax_bh    = fig.add_subplot(gs[1, 0])
    ax_sch   = fig.add_subplot(gs[1, 1])
    ax_ext   = None


def forest_plot(ax, names, hr_arr, lo_arr, hi_arr, p_arr,
                title, color="#457B9D", label_dict=None):
    if label_dict is None:
        label_dict = LABELS
    nn = len(names)
    y  = np.arange(nn)[::-1]
    for i, (yi, col) in enumerate(zip(y, names)):
        hr, lo, hi, pv = hr_arr[i], lo_arr[i], hi_arr[i], p_arr[i]
        c  = "#E63946" if pv < 0.05 else color
        mk = "D" if pv < 0.05 else "o"
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.0, solid_capstyle="round")
        ax.scatter(hr, yi, color=c, s=50, zorder=5, marker=mk)
    ax.axvline(1.0, color="#888", lw=1.2, ls="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [label_dict.get(c, c).split(" [ref")[0] for c in names], fontsize=7.0
    )
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(0.02, 80)
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim()); ax_r.set_yticks(y)
    ax_r.set_yticklabels(
        ["***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
         for pv in p_arr], fontsize=9, color="#E63946")
    for sp in ["top", "right", "left"]:
        ax_r.spines[sp].set_visible(False)
    ax_r.tick_params(axis="y", length=0); ax_r.grid(False)


# ── A. Univariate ─────────────────────────────────────────────────────────────
forest_plot(ax_uni, ALL_NAMES,
            [uni_results[c]["hr"][0]    for c in ALL_NAMES],
            [uni_results[c]["hr_lo"][0] for c in ALL_NAMES],
            [uni_results[c]["hr_hi"][0] for c in ALL_NAMES],
            [uni_results[c]["p"][0]     for c in ALL_NAMES],
            "A.  Univariate Cox Models\n(model-based 95% CIs)",
            color="#457B9D")

# ── B. Non-PH covariates from the extended model ──────────────────────────────
# Panel B shows only the covariates that satisfy the PH assumption.  Because PH
# violations bias all coefficients, we pull these estimates from the *extended*
# model (ext) rather than the main model (multi).  Each non-PH covariate has a
# single, time-constant hazard ratio that can be read directly off the forest
# plot.  PH-violating covariates are shown in Panel E with their interaction
# term, because their effect depends on time.
if ext is not None:
    non_violators  = [c for c in ALL_NAMES if c not in violators]
    non_viol_idx   = [list(EXT_ALL).index(c) for c in non_violators]
    forest_plot(
        ax_multi, non_violators,
        ext["hr"][non_viol_idx], ext["hr_lo"][non_viol_idx],
        ext["hr_hi"][non_viol_idx], ext["p"][non_viol_idx],
        "B.  Non-PH Covariates — Extended Model\n(cluster-robust 95% CIs)",
        color="#2A9D8F",
    )
else:
    # No violations: main model coefficients are unbiased, show all covariates.
    forest_plot(ax_multi, ALL_NAMES,
                multi["hr"], multi["hr_lo"], multi["hr_hi"], multi["p"],
                "B.  Multivariable Cox Model\n(cluster-robust 95% CIs)",
                color="#2A9D8F")

# ── C. Baseline cumulative hazard ─────────────────────────────────────────────
b_times, H0 = breslow_baseline(multi_std["model"], X_std)
ax_bh.step(b_times / 365.25, H0, where="post", color=PALETTE[0], lw=2.2)
ax_bh.fill_between(b_times / 365.25, 0, H0, step="post",
                   alpha=0.12, color=PALETTE[0])
ax_bh.set_title("C.  Breslow Baseline Cumulative Hazard  H₀(t)",
                fontsize=12, fontweight="bold", pad=10)
ax_bh.set_xlabel("Time (years)", fontsize=10)
ax_bh.set_ylabel("H₀(t)", fontsize=10)
ax_bh.set_xlim(left=0); ax_bh.set_ylim(bottom=0)

# ── D. Schoenfeld residuals — PH-violating variables only ────────────────────
# Show the variables that *failed* the PH test: these are the ones the plot is
# meant to diagnose.  Cap at 12 for readability.
show_cols  = violators[:12] if len(violators) > 12 else violators
colors_sch = [PALETTE[i % len(PALETTE)] for i in range(len(show_cols))]
ax_sch.set_title(
    "D.  Schoenfeld Residuals  (PH Assumption Check)\n"
    f"Showing {len(show_cols)} PH-violating variables  (p < {PH_ALPHA})",
    fontsize=12, fontweight="bold", pad=10,
)
for col, c in zip(show_cols, colors_sch):
    if col not in ALL_NAMES:
        continue
    k       = ALL_NAMES.index(col)
    r       = resids[:, k]
    rho, pv = ph_tests[col]
    ord_t   = np.argsort(e_times)
    ts      = e_times[ord_t] / 365.25
    rs      = r[ord_t]
    w       = max(1, len(rs) // 10)
    smooth  = np.convolve(rs, np.ones(w) / w, mode="valid")
    tsm     = ts[w//2: w//2 + len(smooth)]
    ax_sch.scatter(ts, rs, alpha=0.25, s=12, color=c)
    lbl = (f"{LABELS[col].split(' [ref')[0].split('  ×')[0]}"
           f"  ρ={rho:.2f}, p={pv:.3f} !")
    ax_sch.plot(tsm, smooth, color=c, lw=2.0, label=lbl)
ax_sch.axhline(0, color="#888", lw=1, ls="--")
ax_sch.set_xlabel("Time (years)", fontsize=10)
ax_sch.set_ylabel("Schoenfeld residual", fontsize=10)
ax_sch.legend(fontsize=7.5, loc="upper right", framealpha=0.85, edgecolor="#ccc")

# ── E. PH-violating covariates: base coefficient + time interaction ───────────
# Each PH-violating variable requires *two* rows — the base term (β₁) and the
# interaction term (β₂) — because the full effect is β₁ + β₂·log(t).
# Reading β₁ alone is misleading: it is only the log-HR when log(t) = 0,
# i.e. t = 1 day (given the +1 offset).  Both rows are shown together so the
# reader can evaluate the direction and magnitude of time-dependence.
if ax_ext is not None and ext is not None:
    # Build ordered list: base term then interaction term for each violator
    ext_display_names = []
    for v in violators:
        ext_display_names.append(v)              # β₁: base coefficient
        ext_display_names.append(f"{v}:log(t)")  # β₂: time-interaction

    ext_display_idx = [list(EXT_ALL).index(c) for c in ext_display_names]

    nn_ext = len(ext_display_names)
    y_pos  = np.arange(nn_ext)[::-1]

    # Draw a faint horizontal band behind each pair for visual grouping
    for pair_i in range(len(violators)):
        # two rows per pair; top row index in y_pos is pair_i*2, bottom is pair_i*2+1
        y_top    = y_pos[pair_i * 2]
        y_bot    = y_pos[pair_i * 2 + 1]
        band_col = "#e8e8e0" if pair_i % 2 == 0 else "#FAFAF7"
        ax_ext.axhspan(y_bot - 0.45, y_top + 0.45,
                       color=band_col, zorder=0, lw=0)

    hr_d  = ext["hr"][ext_display_idx]
    lo_d  = ext["hr_lo"][ext_display_idx]
    hi_d  = ext["hr_hi"][ext_display_idx]
    p_d   = ext["p"][ext_display_idx]

    for i, (yi, col) in enumerate(zip(y_pos, ext_display_names)):
        hr, lo, hi, pv = hr_d[i], lo_d[i], hi_d[i], p_d[i]
        is_tv  = col in EXT_NAMES          # True for interaction rows
        color  = "#6A4C93" if is_tv else "#F4A261"
        mk     = "s" if is_tv else "D" if pv < 0.05 else "o"
        ax_ext.plot([lo, hi], [yi, yi], color=color, lw=2.0, solid_capstyle="round")
        ax_ext.scatter(hr, yi, color=color, s=50, zorder=5, marker=mk)

    ax_ext.axvline(1.0, color="#888", lw=1.2, ls="--", zorder=0)
    ax_ext.set_yticks(y_pos)
    ax_ext.set_yticklabels(
        [FULL_LABELS.get(c, c).split(" [ref")[0] for c in ext_display_names],
        fontsize=7.0,
    )
    ax_ext.set_xlabel("Hazard Ratio (95% CI)", fontsize=10)
    ax_ext.set_xscale("log")
    ax_ext.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_ext.set_xlim(0.02, 80)
    ax_ext.set_title(
        f"E.  PH-Violating Covariates — Base Coefficient (●/◆) + Time Interaction (■)  [β₁ + β₂·log(t)]\n"
        f"Orange = base term β₁  (log-HR at t = 1 day);  Purple = interaction β₂  (log-HR change per unit log(t))  |  "
        f"Pairs shaded together",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax_ext_r = ax_ext.twinx()
    ax_ext_r.set_ylim(ax_ext.get_ylim())
    ax_ext_r.set_yticks(y_pos)
    ax_ext_r.set_yticklabels(
        ["***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
         for pv in p_d], fontsize=9, color="#E63946",
    )
    for sp in ["top", "right", "left"]:
        ax_ext_r.spines[sp].set_visible(False)
    ax_ext_r.tick_params(axis="y", length=0)
    ax_ext_r.grid(False)
elif ax_ext is not None:
    ax_ext.axis("off")

plt.savefig("baseline_results_full.png", dpi=180, bbox_inches="tight", facecolor="#FAFAF7")
print("Figure saved → baseline_results_full.png")
plt.close()