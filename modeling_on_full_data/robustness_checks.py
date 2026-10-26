"""
Robustness checks for the intrastate peace-agreement Cox model.

Two reviewer concerns:
  1. Region dummies dominate LASSO, so negotiators' provisions may have
     little association with duration once geography is held fixed.
  2. Dropping interstate conflicts and incomplete rows (374 in PAD v22.1
     down to 336) may induce selection bias.

Checks:
  * Region-stratified Cox model for the provisions highlighted in the paper,
    with conflict-clustered standard errors, plus a joint Wald test.
  * Within-region LASSO: covariates are demeaned inside region before the
    penalty is applied, so selection uses only within-region differences.
  * The reported 52-predictor fit at lambda 0.040, ranked by absolute
    standardized coefficient, the 1-SE subset, and 200 bootstrap intervals
    at that penalty. One resample draws agreements. Another draws whole
    conflicts. Each interval is reported raw and among the non-zero draws,
    because a LASSO zero is an HR of exactly 1.
  * Sample sensitivity: re-estimate the stratified model after restoring
    interstate agreements, and bound the six intrastate agreements whose
    durations cannot be computed (immediate failure vs. censored on
    2021-12-31).

Inputs:
  robustness_data/pad_v221_model_frame.csv
    UCDP PAD v22.1 columns used by the model, plus an interstate flag
    joined from UCDP/PRIO Armed Conflict Dataset v22.1 (type_of_conflict == 2).

Run from this directory:
  python robustness_checks.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.stats import chi2
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.model_selection import KFold

np.random.seed(42)

CENSOR = pd.Timestamp("2021-12-31")
DATA = str(Path(__file__).resolve().parent / "robustness_data" / "pad_v221_model_frame.csv")

BINARY = [
    "cease", "intarmy", "ddr", "withd", "pp", "intgov", "intciv",
    "elections", "interim", "natalks", "aut", "fed", "ind", "ref",
    "shaloc", "regdev", "cul", "demarcation", "locgov", "amn", "pris",
    "recon", "return", "reaffirm", "outlin", "pko", "gender", "co_impl",
]
ORDINAL = ["termdur"]
REGION_DUMMIES = [
    "region_europe", "region_middleeast", "region_asia", "region_americas",
]
OTHER_DUMMIES = [
    "incomp_gov", "incomp_both",
    "patype_partial", "patype_process",
    "frame_frame2", "frame_frame3",
]
INTERACTIONS = [
    ("cease", "pko"),
    ("ddr", "intarmy"),
    ("ddr", "intgov"),
    ("pp", "elections"),
    ("amn", "co_impl"),
    ("shaloc", "return"),
    ("aut", "shaloc"),
    ("withd", "pko"),
    ("reaffirm", "co_impl"),
    ("demarcation", "locgov"),
    ("pris", "amn"),
    ("amn", "recon"),
    ("intgov", "natalks"),
]
# Provisions the manuscript reports individually.
HEADLINE = ["reaffirm", "ref", "natalks", "cease", "amn", "recon", "amn_x_recon"]


def _as_code(value):
    """Map PAD incompatibility codes, including the rare '1, 2' strings, to int."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if "," in text:
        parts = {int(float(p)) for p in text.split(",")}
        if parts == {1, 2}:
            return 3
        raise ValueError(f"unparsed code {value!r}")
    return int(float(text))


def load_frame(path=DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pa_date"] = pd.to_datetime(df["pa_date"])
    df["end_date"] = pd.to_datetime(df["duration"], errors="coerce")
    df["event_obs"] = df["ended"].fillna(0).eq(1)
    # Match the main pipeline: termdur code -99 is stored as 0.
    df["termdur"] = df["termdur"].replace(-99, 0).astype(float)
    df["incompatibility"] = df["incompatibility"].map(_as_code)
    # A missing conflict id is its own cluster. Filling it with paid could
    # merge that agreement into a real conflict that happens to share the number.
    ids = df["conflict_id"].to_numpy()
    labels = ids.astype(object)
    for i in np.flatnonzero(pd.isna(ids)):
        labels[i] = f"unclustered-{i}"
    df["cluster"] = labels.astype(str)

    signed = df["pa_date"]
    observed = np.where(
        df["event_obs"],
        (df["end_date"] - signed).dt.days,
        (CENSOR - signed).dt.days,
    )
    df["t_obs"] = observed.astype(float)
    df["valid_time"] = df["t_obs"].notna() & (df["t_obs"] > 0)

    for code, name in [(1, "europe"), (2, "middleeast"), (3, "asia"), (5, "americas")]:
        df[f"region_{name}"] = (df["region"] == code).astype(float)
    for code, name in [(2, "gov"), (3, "both")]:
        df[f"incomp_{name}"] = (df["incompatibility"] == code).astype(float)
    for code, name in [(2, "partial"), (3, "process")]:
        df[f"patype_{name}"] = (df["pa_type"] == code).astype(float)
    for code, name in [(2, "frame2"), (3, "frame3")]:
        df[f"frame_{name}"] = (df["frame"] == code).astype(float)
    return df


def analysis_mask(df: pd.DataFrame) -> pd.Series:
    """Intrastate agreements with a usable duration. This is the paper's n=336."""
    return (~df["interstate"].astype(bool)) & df["valid_time"]


def partial_loglik(beta, X, t, e):
    """Breslow partial log-likelihood, scaled by the number of events."""
    order = np.argsort(t)
    X_s, t_s, e_s = X[order], t[order], e[order]
    eta = X_s @ beta
    eta -= eta.max()
    exp_s = np.exp(eta)
    cum = np.cumsum(exp_s[::-1])[::-1]
    ll = 0.0
    for i in range(len(t_s)):
        if e_s[i]:
            ll += eta[i] - np.log(cum[i] + 1e-300)
    return ll / (e_s.sum() + 1e-300)


def design_matrix(df, columns, within_region=False):
    """Standardized design matrix. Interactions are residualized on their mains.

    within_region=True subtracts the region mean from every non-region column
    before standardization, so later coefficients use within-region variation.
    """
    base_cols = [c for c in columns if "::" not in c]
    X_raw = df[base_cols].to_numpy(float)
    if within_region:
        region = df["region"].to_numpy()
        for r in np.unique(region):
            rows = region == r
            X_raw[rows] -= X_raw[rows].mean(axis=0)
    sd = X_raw.std(axis=0)
    sd[sd == 0] = 1.0
    X_std = (X_raw - X_raw.mean(axis=0)) / sd

    pieces = [X_std]
    sds = [sd]
    names = list(base_cols)
    n = len(df)
    for a, b in INTERACTIONS:
        name = f"{a}::{b}"
        if name not in columns:
            continue
        ia, ib = names.index(a), names.index(b)
        product = X_std[:, ia] * X_std[:, ib]
        Z = np.column_stack([np.ones(n), X_std[:, ia], X_std[:, ib]])
        coef, *_ = lstsq(Z, product, rcond=None)
        resid = product - Z @ coef
        rsd = resid.std() or 1.0
        pieces.append(((resid - resid.mean()) / rsd)[:, None])
        sds.append(np.array([rsd]))
        names.append(name)
    X = np.hstack(pieces)
    scale = np.concatenate(sds)
    # Drop columns that are constant after within-region demeaning.
    vary = X.std(axis=0) > 1e-8
    return X[:, vary], scale[vary], [names[i] for i in np.where(vary)[0]]


def lasso_cv(X, t, e):
    """5-fold Verweij–van Houwelingen CV, matching the main LASSO script."""
    y = Surv.from_arrays(event=e.astype(bool), time=t)
    path = CoxnetSurvivalAnalysis(
        l1_ratio=1.0, n_alphas=100, alpha_min_ratio=0.01,
        normalize=False, fit_baseline_model=False, max_iter=100_000, tol=1e-8,
    )
    path.fit(X, y)
    alphas = path.alphas_
    e = e.astype(float)
    scores = np.zeros((5, len(alphas)))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for k, (tr, va) in enumerate(kf.split(X)):
        m = CoxnetSurvivalAnalysis(
            l1_ratio=1.0, alphas=alphas, normalize=False,
            fit_baseline_model=False, max_iter=100_000, tol=1e-7,
        )
        m.fit(X[tr], y[tr])
        n_fit = m.coef_.shape[1]
        for j, _alpha in enumerate(alphas):
            beta = m.coef_[:, min(j, n_fit - 1)]
            ll_full = partial_loglik(beta, X, t, e) * e.sum()
            ll_tr = partial_loglik(beta, X[tr], t[tr], e[tr]) * e[tr].sum()
            scores[k, j] = (ll_full - ll_tr) / max(e[va].sum(), 1)
    mean = scores.mean(axis=0)
    se = scores.std(axis=0) / np.sqrt(5)
    best = int(np.argmax(mean))
    one_se = int(np.where(mean >= mean[best] - se[best])[0][0])
    refit = CoxnetSurvivalAnalysis(
        l1_ratio=1.0, alphas=[float(alphas[best])], normalize=False,
        fit_baseline_model=False, max_iter=200_000, tol=1e-10,
    )
    refit.fit(X, y)
    beta_std = refit.coef_.ravel()
    n_fit = path.coef_.shape[1]
    return {
        "alpha": float(alphas[best]),
        "alpha_1se": float(alphas[one_se]),
        "cv": float(mean[best]),
        "beta_std": beta_std,
        "n_selected": int(np.sum(np.abs(beta_std) > 1e-6)),
        "alpha_1se_index": one_se,
        "alphas": np.asarray(alphas[:n_fit], float),
        "coef_path": np.asarray(path.coef_[:, :n_fit], float),
    }


def _fit_penalized(X, y, lam, idx):
    yb = y[idx]
    if int(yb["event"].sum()) < 5:
        return None, "fewer than 5 events"
    m = CoxnetSurvivalAnalysis(
        l1_ratio=1.0, alphas=[float(lam)], normalize=False,
        fit_baseline_model=False, max_iter=100_000, tol=1e-7,
    )
    try:
        m.fit(X[idx], yb)
    except Exception as err:
        return None, f"{type(err).__name__}: {err}"
    return m.coef_.ravel(), None


def bootstrap_coefs(X, y, lam, draw, B=200):
    """Fixed-penalty LASSO replicates. Failed fits stay missing.

    draw(b) returns the row index for replicate b. Pass an agreement-level
    draw or a draw that resamples whole conflicts.
    """
    boots = np.full((B, X.shape[1]), np.nan)
    n_fail = 0
    first_fail = None
    for b in range(B):
        coef, err = _fit_penalized(X, y, lam, draw(b))
        if coef is None:
            n_fail += 1
            first_fail = first_fail or err
        else:
            boots[b] = coef
    return boots, n_fail, first_fail


def hr_interval(coef_std, scale):
    """Percentile HR interval, and the same interval among non-zero draws.

    A LASSO replicate of exactly zero is an HR of 1, so the raw percentile
    often lands on 1 even when the selected replicates do not.
    """
    coef_std = np.asarray(coef_std, float)
    ok = np.isfinite(coef_std)
    selected = ok & (np.abs(coef_std) > 1e-8)
    hr = np.exp(coef_std / scale)
    n_ok = int(ok.sum())
    n_sel = int(selected.sum())
    out = {
        "lo": float(np.nanpercentile(hr, 2.5)),
        "hi": float(np.nanpercentile(hr, 97.5)),
        "zero_share": float(1 - n_sel / n_ok) if n_ok else np.nan,
        "n_selected_draws": n_sel,
        "lo_selected": np.nan,
        "hi_selected": np.nan,
    }
    if n_sel >= 20:
        kept = hr[selected]
        out["lo_selected"] = float(np.percentile(kept, 2.5))
        out["hi_selected"] = float(np.percentile(kept, 97.5))
    return out


def cluster_draw(labels, rng):
    """Resample whole conflicts, then return the stacked row index."""
    labels = np.asarray(labels)
    members = [np.flatnonzero(labels == u) for u in pd.unique(labels)]

    def draw(_b):
        chosen = rng.integers(0, len(members), size=len(members))
        return np.concatenate([members[int(i)] for i in chosen])

    return draw, len(members)


def reported_operating_point(fit, scale, names, X, y):
    """Coefficients at lambda 0.040 and at the 1-SE penalty.

    Ranking uses the absolute standardized coefficient. Hazard ratios are
    exp(beta) after dividing by the column standard deviation.
    """
    alphas = fit["alphas"]
    at = int(np.argmin(np.abs(alphas - 0.040)))
    beta_std = fit["coef_path"][:, at]
    beta = beta_std / scale
    order = np.argsort(-np.abs(beta_std))
    rows = []
    for i in order:
        if abs(beta_std[i]) <= 1e-6:
            continue
        rows.append({
            "term": names[i],
            "beta_std": float(beta_std[i]),
            "beta_orig": float(beta[i]),
            "hr": float(np.exp(beta[i])),
        })
    m1 = CoxnetSurvivalAnalysis(
        l1_ratio=1.0, alphas=[float(fit["alpha_1se"])], normalize=False,
        fit_baseline_model=False, max_iter=200_000, tol=1e-10,
    )
    m1.fit(X, y)
    keep = np.abs(m1.coef_.ravel()) > 1e-6
    one_se_terms = [names[i] for i in np.argsort(-np.abs(m1.coef_.ravel())) if keep[i]]
    return {
        "lambda_reported": float(alphas[at]),
        "n_reported": len(rows),
        "rows": rows,
        "one_se_terms": one_se_terms,
    }


def selected_table(names, scale, beta_std):
    beta = beta_std / scale
    rows = []
    for name, b, bs in zip(names, beta, beta_std):
        if abs(bs) > 1e-6:
            rows.append((name, b, np.exp(b)))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return rows


def fit_clustered_cox(df, covariates, stratify_region=True):
    cols = ["t", "event", "cluster"] + covariates
    data = df[cols].copy()
    # Patsy treats ':' as an interaction operator, so rename residual products.
    data = data.rename(columns={"amn::recon": "amn_x_recon"})
    covariates = ["amn_x_recon" if c == "amn::recon" else c for c in covariates]
    if stratify_region:
        data["region"] = df["region"].to_numpy()
    cph = CoxPHFitter()
    kwargs = dict(
        duration_col="t",
        event_col="event",
        cluster_col="cluster",
        robust=True,
    )
    if stratify_region:
        kwargs["strata"] = ["region"]
    cph.fit(data, **kwargs)
    return cph


def wald_joint(cph, names):
    present = [n for n in names if n in cph.params_.index]
    idx = [cph.params_.index.get_loc(n) for n in present]
    b = cph.params_.to_numpy()[idx]
    V = cph.variance_matrix_.to_numpy()[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(V, b))
    p = float(chi2.sf(stat, len(present)))
    return stat, p, present


def hr_table(cph, names):
    rows = []
    summary = cph.summary
    for name in names:
        if name not in summary.index:
            continue
        r = summary.loc[name]
        rows.append({
            "term": name,
            "hr": float(r["exp(coef)"]),
            "lo": float(r["exp(coef) lower 95%"]),
            "hi": float(r["exp(coef) upper 95%"]),
            "p": float(r["p"]),
        })
    return pd.DataFrame(rows)


def prepare_sample(df, extra_t=None, extra_event=None):
    """Return a modeling frame with t and event columns."""
    out = df.copy()
    out["t"] = out["t_obs"] if extra_t is None else extra_t
    out["event"] = out["event_obs"].astype(bool) if extra_event is None else extra_event
    out = out.loc[out["t"].notna() & (out["t"] > 0)].copy()
    # Residualize the amnesty x reconciliation product on the two mains,
    # on the raw (not standardized) scale so the Cox coefficient is an HR
    # for the residual interaction in the same units as the main LASSO.
    a = out["amn"].to_numpy(float)
    b = out["recon"].to_numpy(float)
    a = (a - a.mean()) / (a.std() or 1)
    b = (b - b.mean()) / (b.std() or 1)
    product = a * b
    Z = np.column_stack([np.ones(len(out)), a, b])
    coef, *_ = lstsq(Z, product, rcond=None)
    resid = product - Z @ coef
    out["amn::recon"] = (resid - resid.mean()) / (resid.std() or 1)
    return out


def ph_flags(X, names, t, e, alpha=0.05):
    """Grambsch-Therneau test on log time. Returns names with p < alpha."""
    from lifelines.statistics import proportional_hazard_test

    safe = [n.replace("::", "_x_") for n in names]
    frame = pd.DataFrame(X, columns=safe)
    frame["t"] = np.asarray(t, float)
    frame["e"] = np.asarray(e, int)
    cph = CoxPHFitter()
    cph.fit(frame, duration_col="t", event_col="e")
    summary = proportional_hazard_test(cph, frame, time_transform="log").summary
    bad = summary.loc[summary["p"] < alpha].sort_values("p")
    back = dict(zip(safe, names))
    return [(back[ix], float(bad.loc[ix, "p"])) for ix in bad.index]


def natalks_log_time(X, names, t, e):
    """Counting-process Cox with a national-dialogue x log(t) term.

    Returns the main-effect coefficient, the log-time slope, its p-value,
    and the follow-up time in years where the linear predictor crosses 0.
    """
    from lifelines import CoxTimeVaryingFitter

    t = np.asarray(t, float)
    e = np.asarray(e, bool)
    j = names.index("natalks")
    safe = [n.replace("::", "_x_") for n in names]
    pieces = []
    prev = 0.0
    for tk in np.unique(t[e]):
        mask = t > prev
        stop = np.minimum(t[mask], tk)
        keep = stop > prev
        idx = np.flatnonzero(mask)[keep]
        stop = stop[keep]
        block = {safe[k]: X[idx, k] for k in range(X.shape[1])}
        block["start"] = np.full(len(idx), prev)
        block["stop"] = stop
        block["event"] = ((t[idx] == tk) & e[idx]).astype(int)
        block["natalks_logt"] = X[idx, j] * np.log(stop)
        block["id"] = idx
        pieces.append(pd.DataFrame(block))
        prev = float(tk)
    long = pd.concat(pieces, ignore_index=True)
    fit = CoxTimeVaryingFitter(penalizer=0.0)
    fit.fit(long, id_col="id", start_col="start", stop_col="stop", event_col="event", show_progress=False)
    b1 = float(fit.params_["natalks"])
    b2 = float(fit.params_["natalks_logt"])
    p_slope = float(fit.summary.loc["natalks_logt", "p"])
    cross_years = float(np.exp(-b1 / b2) / 365.25) if b2 != 0 else np.nan
    return b1, b2, p_slope, cross_years


def fmt_hr(row):
    return f"HR {row['hr']:.2f} (95% CI {row['lo']:.2f}–{row['hi']:.2f}, p={row['p']:.3f})"


def main():
    raw = load_frame()
    base_mask = analysis_mask(raw)
    base = raw.loc[base_mask].copy()
    print("=" * 72)
    print("SAMPLE ACCOUNTING")
    print("=" * 72)
    n_pad = len(raw)
    n_inter = int(raw["interstate"].sum())
    intra = raw.loc[~raw["interstate"].astype(bool)]
    missing_time = intra.loc[~intra["valid_time"]]
    print(f"PAD v22.1 agreements: {n_pad}")
    print(f"Interstate (UCDP type 2 only): {n_inter}")
    print(f"Intrastate with usable duration: {len(base)}  "
          f"events={int(base['event_obs'].sum())}  "
          f"censored={int((~base['event_obs']).sum())}")
    print(f"Intrastate without usable duration: {len(missing_time)}")
    print(missing_time[["paid", "conflict_name", "event_obs", "t_obs", "reaffirm"]].to_string(index=False))
    print("Region counts (agreements / events):")
    for code, label in [(1, "Europe"), (2, "Middle East"), (3, "Asia"), (4, "Africa"), (5, "Americas")]:
        sub = base.loc[base["region"] == code]
        print(f"  {label:<12} n={len(sub):<3} events={int(sub['event_obs'].sum())}")

    # Headline region-stratified Cox on the estimation sample.
    model_df = prepare_sample(base)
    controls = OTHER_DUMMIES + ORDINAL
    covariates = ["reaffirm", "ref", "natalks", "cease", "amn", "recon", "amn::recon"] + controls
    print("\n" + "=" * 72)
    print("REGION-STRATIFIED COX (conflict-clustered SEs)")
    print("=" * 72)
    cph = fit_clustered_cox(model_df, covariates, stratify_region=True)
    table = hr_table(cph, HEADLINE)
    print(table.to_string(index=False))
    stat, p, used = wald_joint(cph, HEADLINE)
    print(f"Joint Wald on {used}: chi2={stat:.2f}, df={len(used)}, p={p:.4g}")

    # Within-region LASSO and full LASSO, for the predictive comparison.
    full_cols = BINARY + ORDINAL + REGION_DUMMIES + OTHER_DUMMIES
    # design_matrix appends interactions listed in INTERACTIONS when both mains exist
    X_full, scale_full, names_full = design_matrix(base, full_cols + [f"{a}::{b}" for a, b in INTERACTIONS])
    within_cols = BINARY + ORDINAL + OTHER_DUMMIES
    X_within, scale_within, names_within = design_matrix(
        base, within_cols + [f"{a}::{b}" for a, b in INTERACTIONS], within_region=True,
    )
    t = base["t_obs"].to_numpy(float)
    e = base["event_obs"].to_numpy(bool)
    print("\n" + "=" * 72)
    print("LASSO COMPARISON")
    print("=" * 72)
    print("Fitting full LASSO...")
    full = lasso_cv(X_full, t, e)
    full["beta"] = full["beta_std"] / scale_full
    print("Fitting within-region LASSO...")
    within = lasso_cv(X_within, t, e)
    within["beta"] = within["beta_std"] / scale_within
    # one-SE counts
    print(f"Full LASSO: lambda_min={full['alpha']:.4f}, selected={full['n_selected']}, "
          f"CV partial LL={full['cv']:.4f}, p={X_full.shape[1]}")
    print(f"Within-region LASSO: lambda_min={within['alpha']:.4f}, "
          f"selected={within['n_selected']}, CV partial LL={within['cv']:.4f}, "
          f"p={X_within.shape[1]}")
    print("\nWithin-region non-zero coefficients (original scale):")
    for name, beta, hr in selected_table(names_within, scale_within, within["beta_std"]):
        print(f"  {name:<28} beta={beta:+.3f}  HR={hr:.3f}")

    reported = reported_operating_point(full, scale_full, names_full, X_full, Surv.from_arrays(event=e, time=t))
    print("\n" + "=" * 72)
    print(f"REPORTED LASSO  lambda={reported['lambda_reported']:.4f}  "
          f"selected={reported['n_reported']}  (ranked by |standardized beta|)")
    print("=" * 72)
    for row in reported["rows"]:
        print(f"  {row['term']:<22} std={row['beta_std']:+.3f}  "
              f"beta={row['beta_orig']:+.3f}  HR={row['hr']:.3f}")
    print(f"\n1-SE penalty lambda={full['alpha_1se']:.4f}  "
          f"selected={len(reported['one_se_terms'])}")
    print("  " + ", ".join(reported["one_se_terms"]))

    y_boot = Surv.from_arrays(event=e, time=t)
    lam = reported["lambda_reported"]
    rs = np.random.RandomState(42)
    n_rows = len(X_full)

    def draw_agreements(_b):
        return rs.choice(n_rows, n_rows, replace=True)

    print("\nAgreement-level bootstrap at the reported penalty (B=200)...")
    boots, n_fail, first_fail = bootstrap_coefs(X_full, y_boot, lam, draw_agreements)
    print(f"  fits {boots.shape[0] - n_fail}/{boots.shape[0]}")
    if first_fail:
        print(f"  first failed replicate: {first_fail}")
    draw_conflicts, n_clusters = cluster_draw(base["cluster"].to_numpy(), np.random.default_rng(42))
    print(f"Conflict-level bootstrap at the reported penalty (B=200, G={n_clusters})...")
    boots_cl, n_fail_cl, first_fail_cl = bootstrap_coefs(X_full, y_boot, lam, draw_conflicts)
    print(f"  fits {boots_cl.shape[0] - n_fail_cl}/{boots_cl.shape[0]}")
    if first_fail_cl:
        print(f"  first failed replicate: {first_fail_cl}")
    boot_rows = []
    for row in reported["rows"]:
        i = names_full.index(row["term"])
        agree = hr_interval(boots[:, i], scale_full[i])
        cluster = hr_interval(boots_cl[:, i], scale_full[i])
        row.update({
            "boot_lo": agree["lo"],
            "boot_hi": agree["hi"],
            "boot_includes_1": bool(agree["lo"] <= 1 <= agree["hi"]),
            "boot_zero_share": agree["zero_share"],
            "boot_lo_selected": agree["lo_selected"],
            "boot_hi_selected": agree["hi_selected"],
            "cluster_lo": cluster["lo"],
            "cluster_hi": cluster["hi"],
            "cluster_includes_1": bool(cluster["lo"] <= 1 <= cluster["hi"]),
            "cluster_zero_share": cluster["zero_share"],
        })
        boot_rows.append(row)
        agree_flag = "includes 1" if row["boot_includes_1"] else "excludes 1"
        cluster_flag = "includes 1" if row["cluster_includes_1"] else "excludes 1"
        print(f"  {row['term']:<22} HR={row['hr']:.3f}  "
              f"zero={row['boot_zero_share']:.0%}  "
              f"agreement [{agree['lo']:.3f}, {agree['hi']:.3f}] {agree_flag}  "
              f"selected [{agree['lo_selected']:.3f}, {agree['hi_selected']:.3f}]  "
              f"conflict [{cluster['lo']:.3f}, {cluster['hi']:.3f}] {cluster_flag}")

    # Sample sensitivity on the same stratified specification.
    print("\n" + "=" * 72)
    print("SAMPLE SENSITIVITY")
    print("=" * 72)
    interstate = raw.loc[raw["interstate"].astype(bool) & raw["valid_time"]].copy()
    print(f"Interstate with usable duration: {len(interstate)}  "
          f"events={int(interstate['event_obs'].sum())}")
    pooled = pd.concat([base, interstate], ignore_index=True)
    pooled_df = prepare_sample(pooled)
    pooled_df["interstate"] = pooled["interstate"].to_numpy(float)
    cph_pool = fit_clustered_cox(
        pooled_df, covariates + ["interstate"], stratify_region=True,
    )
    pool_table = hr_table(cph_pool, HEADLINE + ["interstate"])

    def bound_sample(mode):
        extra = missing_time.copy()
        if mode == "fail":
            extra["t_obs"] = 1.0
            extra["event_obs"] = True
        else:
            extra["t_obs"] = (CENSOR - extra["pa_date"]).dt.days.astype(float)
            extra["event_obs"] = False
        glued = pd.concat([base, extra], ignore_index=True)
        return prepare_sample(glued)

    cph_fail = fit_clustered_cox(bound_sample("fail"), covariates, stratify_region=True)
    cph_cens = fit_clustered_cox(bound_sample("censor"), covariates, stratify_region=True)
    fail_table = hr_table(cph_fail, HEADLINE)
    cens_table = hr_table(cph_cens, HEADLINE)

    merged = table.merge(pool_table, on="term", suffixes=("_base", "_pool"))
    merged = merged.merge(fail_table.add_suffix("_fail").rename(columns={"term_fail": "term"}), on="term")
    merged = merged.merge(cens_table.add_suffix("_cens").rename(columns={"term_cens": "term"}), on="term")
    show = merged[["term", "hr_base", "hr_pool", "hr_fail", "hr_cens", "p_base", "p_pool"]]
    print(show.to_string(index=False))
    stat_p, p_p, _ = wald_joint(cph_pool, HEADLINE)
    print(f"Pooled joint Wald p={p_p:.4g} (chi2={stat_p:.2f})")

    # Provision balance: analysis sample vs interstate vs undated intrastate.
    print("\nProvision means (analysis / interstate / undated intrastate):")
    for col in ["reaffirm", "ref", "natalks", "cease", "amn", "recon"]:
        print(f"  {col:<12} {base[col].mean():.3f}  "
              f"{interstate[col].mean():.3f}  {missing_time[col].mean():.3f}")

    out = show.copy()
    out["joint_wald_p_base"] = p
    out["joint_wald_p_pool"] = p_p
    out["within_lasso_selected"] = within["n_selected"]
    out["full_lasso_lambda"] = full["alpha"]
    out["full_lasso_selected"] = full["n_selected"]
    out_path = Path(__file__).resolve().parent / "robustness_data" / "robustness_summary.csv"
    out.to_csv(out_path, index=False)
    reported_path = Path(__file__).resolve().parent / "robustness_data" / "reported_lasso.csv"
    reported_df = pd.DataFrame(boot_rows)
    reported_df["lambda_reported"] = reported["lambda_reported"]
    reported_df["lambda_1se"] = full["alpha_1se"]
    reported_df["in_1se"] = reported_df["term"].isin(reported["one_se_terms"])
    reported_df.to_csv(reported_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"Wrote {reported_path}")
    print("\nGrambsch-Therneau (log time) on the 52-column model:")
    try:
        flagged = ph_flags(X_full, names_full, t, e)
    except Exception as err:
        print(f"  test failed: {type(err).__name__}: {err}")
    else:
        if not flagged:
            print("  no predictor with p < 0.05")
        for name, pval in flagged:
            print(f"  {name:<22} p={pval:.4f}")
        print(f"  {len(flagged)} rejections; about {0.05 * len(names_full):.1f} expected by chance at p < 0.05")
        print("  No time-varying terms were added; reported HRs are averages over follow-up.")
    try:
        b1, b2, p_slope, cross_years = natalks_log_time(X_full, names_full, t, e)
    except Exception as err:
        print(f"  national-dialogue log(t) fit failed: {type(err).__name__}: {err}")
    else:
        print(f"  national dialogue log(t): beta={b1:+.3f}, slope={b2:+.3f}, "
              f"p={p_slope:.3f}, crosses 0 at {cross_years:.1f} years")


if __name__ == "__main__":
    main()
