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
#     For each violating covariate x_j, we add a new column:
#         x_j_std_i  *  log(t_i)
#     where t_i is observation i's own event/censoring time.
#     Using the subject's own time is the standard Grambsch-Therneau
#     approach; there is no leakage because log(t) is not a future outcome
#     but a property of the observation's time axis.
#
#     The extended model estimates:
#         beta(t) = beta_1  +  beta_2 * log(t)
#     where beta_1 is the time-averaged effect and beta_2 captures
#     how the effect changes over the log-time scale.
# ═══════════════════════════════════════════════════════════════════════════════

EXT_NAMES  = []    # names of the added x*log(t) columns
EXT_LABELS = {}

if violators:
    log_t = np.log(t_all + 1)     # +1 guards against t=0
    tv_cols = []
    for col in violators:
        j      = ALL_NAMES.index(col)
        tv_col = X_std[:, j] * log_t
        tv_cols.append(tv_col)
        tname  = f"{col}:log(t)"
        EXT_NAMES.append(tname)
        EXT_LABELS[tname] = f"{LABELS[col].split(' [ref')[0]}  ×  log(t)"

    X_ext     = np.hstack([X_std, np.column_stack(tv_cols)])
    EXT_ALL   = ALL_NAMES + EXT_NAMES

    # Standardise only the new tv columns (x_std * log(t) is not unit-free)
    tv_arr    = np.column_stack(tv_cols)
    tv_mean   = tv_arr.mean(axis=0)
    tv_sd     = tv_arr.std(axis=0); tv_sd[tv_sd == 0] = 1.0
    X_ext_std = np.hstack([X_std, (tv_arr - tv_mean) / tv_sd])
    EXT_SD    = np.concatenate([ALL_SD, tv_sd])

    y_ext = Surv.from_arrays(event=event_bool, time=t_all)

    print(f"\nFitting extended model  "
          f"({len(violators)} time-varying term(s): "
          f"{', '.join(violators)})  ...", end="", flush=True)
    ext_std = fit_clustered(X_ext_std, y_ext, t_all, event_float, cluster_ids)
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
# 8.  FIGURES
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

# ── Thematic Groupings ────────────────────────────────────────────────────────

THEMATIC_GROUPS = {
    "External & Prior Conflict Context": [
        "neighbor_at_war", "rebel_victory", "government_victory", 
        "unknown_outcome", "log_bd_total", "regime_change", 
        "v2x_polyarchy", "log_v2regdur", "v2x_freexp_altinf", 
        "v2x_liberal", "v2x_partip", "log_gdp_pc", "gdp_growth", 
        "lmtnest", "log_milper"
    ],
    "Agreement Structure & Conflict Typology": [
        "region_europe", "region_middleeast", "region_asia", "region_americas",
        "incomp_gov", "incomp_both", "patype_partial", "patype_process",
        "frame_frame2", "frame_frame3", "termdur"
    ],
    "Security & Enforcement": [
        "cease", "withd", "ddr", "intarmy", "pko"
    ],
    "Political Power-Sharing & Governance": [
        "pp", "intgov", "intciv", "interim", "elections", 
        "natalks", "co_impl", "reaffirm", "outlin"
    ],
    "Territorial & Decentralization": [
        "aut", "fed", "ind", "shaloc", "locgov", "demarcation", "ref", "regdev"
    ],
    "Transitional Justice, Social & Humanitarian": [
        "amn", "pris", "recon", "return", "cul", "gender"
    ],
    "Interaction Terms": INTERACTION_NAMES
}

# ── Base Forest Plot Function ─────────────────────────────────────────────────

def forest_plot(ax, names, hr_arr, lo_arr, hi_arr, p_arr,
                title, color="#457B9D", label_dict=None):
    if label_dict is None:
        label_dict = FULL_LABELS if 'FULL_LABELS' in globals() else LABELS
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
        [label_dict.get(c, c).split(" [ref")[0] for c in names], fontsize=8.5
    )
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    
    # Standardize x-limits for comparability across plots
    ax.set_xlim(0.05, 20)
    
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim()); ax_r.set_yticks(y)
    ax_r.set_yticklabels(
        ["***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
         for pv in p_arr], fontsize=9, color="#E63946")
    for sp in ["top", "right", "left"]:
        ax_r.spines[sp].set_visible(False)
    ax_r.tick_params(axis="y", length=0); ax_r.grid(False)


def plot_grouped_forest(ax, names, results_dict, title, color="#457B9D"):
    """Sorts variables by HR and plots them."""
    # Filter to only variables that exist in the results
    valid_names = [n for n in names if n in results_dict]
    # Sort from lowest HR to highest HR
    sorted_names = sorted(valid_names, key=lambda x: results_dict[x]["hr"][0])
    
    hr_arr = [results_dict[c]["hr"][0] for c in sorted_names]
    lo_arr = [results_dict[c]["hr_lo"][0] for c in sorted_names]
    hi_arr = [results_dict[c]["hr_hi"][0] for c in sorted_names]
    p_arr  = [results_dict[c]["p"][0] for c in sorted_names]
    
    forest_plot(ax, sorted_names, hr_arr, lo_arr, hi_arr, p_arr, title, color=color)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: UNIVARIATE MODELS (THEMATICALLY GROUPED & SORTED)
# ═══════════════════════════════════════════════════════════════════════════════

fig1 = plt.figure(figsize=(20, 24), facecolor="#FAFAF7")
fig1.suptitle(
    "Univariate Cox Models by Thematic Category\n(Sorted by Hazard Ratio)",
    fontsize=16, fontweight="bold", y=0.98, color="#1a1a1a"
)

axes_uni = fig1.subplot_mosaic(
    [["context", "struct"],
     ["security", "political"],
     ["territory", "justice"],
     ["interactions", "."]],
    gridspec_kw={"hspace": 0.40, "wspace": 0.35, "height_ratios": [1.5, 1, 1, 1.3]}
)

plot_grouped_forest(axes_uni["context"], THEMATIC_GROUPS["External & Prior Conflict Context"], uni_results, "External & Prior Conflict Context", "#457B9D")
plot_grouped_forest(axes_uni["struct"], THEMATIC_GROUPS["Agreement Structure & Conflict Typology"], uni_results, "Agreement Structure & Conflict Typology", "#2A9D8F")
plot_grouped_forest(axes_uni["security"], THEMATIC_GROUPS["Security & Enforcement"], uni_results, "Security & Enforcement", "#E9C46A")
plot_grouped_forest(axes_uni["political"], THEMATIC_GROUPS["Political Power-Sharing & Governance"], uni_results, "Political Power-Sharing & Governance", "#F4A261")
plot_grouped_forest(axes_uni["territory"], THEMATIC_GROUPS["Territorial & Decentralization"], uni_results, "Territorial & Decentralization", "#6A4C93")
plot_grouped_forest(axes_uni["justice"], THEMATIC_GROUPS["Transitional Justice, Social & Humanitarian"], uni_results, "Transitional Justice & Social Factors", "#E63946")
plot_grouped_forest(axes_uni["interactions"], THEMATIC_GROUPS["Interaction Terms"], uni_results, "Interaction Terms", "#457B9D")

out_uni = "univariate_forest_plots.png"
fig1.savefig(out_uni, dpi=180, bbox_inches="tight", facecolor="#FAFAF7")
print(f"Saved Univariate grouped plots to {out_uni}")
plt.close(fig1)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: MULTIVARIABLE MODEL & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

n_panels_m = 3 if ext is not None else 2
fig2_h = max(16, 4 + len(ALL_NAMES) * 0.35)
fig2 = plt.figure(figsize=(12, fig2_h), facecolor="#FAFAF7")
fig2.suptitle(
    "Multivariable Cox Model & Diagnostics",
    fontsize=16, fontweight="bold", y=0.99, color="#1a1a1a"
)

if ext is not None:
    gs = fig2.add_gridspec(3, 1, height_ratios=[4.0, 1.5, 2.0], hspace=0.35)
    ax_multi = fig2.add_subplot(gs[0])
    ax_diag  = fig2.add_subplot(gs[1])
    ax_ext   = fig2.add_subplot(gs[2])
else:
    gs = fig2.add_gridspec(2, 1, height_ratios=[4.0, 1.5], hspace=0.35)
    ax_multi = fig2.add_subplot(gs[0])
    ax_diag  = fig2.add_subplot(gs[1])
    ax_ext   = None

# ── Multivariable Forest Plot ──
# (Sorting multivariable by HR is optional, but retaining original grouped order is often preferred here. 
# We will plot using ALL_NAMES to show everything).
hr_multi = [multi["hr"][k] for k in range(len(ALL_NAMES))]
lo_multi = [multi["hr_lo"][k] for k in range(len(ALL_NAMES))]
hi_multi = [multi["hr_hi"][k] for k in range(len(ALL_NAMES))]
p_multi  = [multi["p"][k] for k in range(len(ALL_NAMES))]

forest_plot(ax_multi, ALL_NAMES, hr_multi, lo_multi, hi_multi, p_multi,
            "Multivariable Model (Cluster-Robust CIs)", color="#2A9D8F")

# ── Baseline Cumulative Hazard ──
times_bh, chf_bh = breslow_baseline(multi_std["model"], X_std)
ax_diag.step(times_bh / 365.25, chf_bh, where="post", color="#E63946", lw=2)
ax_diag.set_title("Breslow Baseline Cumulative Hazard", fontsize=11, fontweight="bold")
ax_diag.set_xlabel("Time (years)", fontsize=9)
ax_diag.set_ylabel("Cumulative Hazard", fontsize=9)

# ── Extended Model (if applicable) ──
if ax_ext is not None:
    hr_ext = [ext["hr"][k] for k in range(len(EXT_ALL))]
    lo_ext = [ext["hr_lo"][k] for k in range(len(EXT_ALL))]
    hi_ext = [ext["hr_hi"][k] for k in range(len(EXT_ALL))]
    p_ext  = [ext["p"][k] for k in range(len(EXT_ALL))]

    forest_plot(ax_ext, EXT_ALL, hr_ext, lo_ext, hi_ext, p_ext,
                "Extended Model (Time-Varying Terms)", color="#F4A261")

out_multi = "multivariable_diagnostics.png"
fig2.savefig(out_multi, dpi=180, bbox_inches="tight", facecolor="#FAFAF7")
print(f"Saved Multivariable & Diagnostics to {out_multi}")
plt.close(fig2)