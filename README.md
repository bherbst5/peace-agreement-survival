# Peace Agreement Duration: Survival Analysis

Code for: *[Toward Durable Peace Agreements: Provision Selection and Contextual Risk in Intrastate Conflict Resolution]*

Survival analysis of peace agreement duration. The outcome is time to agreement collapse. Agreements still in force on December 31, 2021 are right-censored.

The repository has two modeling tracks. `modeling_pad_alone` uses the Peace Agreement Dataset (PAD) provisions only. `modeling_on_full_data` adds external context (V-Dem indices, conflict history, and macroeconomic indicators).

Three approaches are implemented:

* Kaplan-Meier curves, with a multivariate log-rank test
* Unpenalized Cox proportional hazards models, with conflict-clustered standard errors and time-varying terms for proportional-hazards violators
* LASSO-penalized Cox models, with 5-fold cross-validation on held-out partial log-likelihood (Verweij-van Houwelingen) and bootstrap confidence intervals

---

## Repository Structure

### 1. Full merged analysis (`modeling_on_full_data`)

67-predictor models on PAD plus external covariates.

| File | Description |
| :--- | :--- |
| `pad_merged.csv` | PAD merged with external macroeconomic and conflict variables |
| `kaplan_meier_full.py` | Stratified Kaplan-Meier curves and log-rank tests |
| `baseline_cox_full.py` | Unpenalized Cox model, proportional-hazards tests, and the extended time-varying model |
| `cox_lasso_full.py` | LASSO-penalized Cox model and bootstrap intervals |
| `robustness_checks.py` | Region-stratified Cox models, a within-region LASSO, and bounds for the dropped agreements |
| `paper_figures.py` | Vector PDFs of the regional Kaplan-Meier curves and the LASSO variable-importance chart |
| `robustness_data/pad_v221_model_frame.csv` | PAD v22.1 model frame, with an interstate flag from UCDP/PRIO ACD v22.1 |
| `robustness_data/robustness_summary.csv` | Stratified-Cox table written by `robustness_checks.py` |
| `robustness_data/reported_lasso.csv` | Coefficients and hazard ratios at \(\lambda = 0.040\), plus agreement-level and conflict-level bootstrap intervals. `cluster_zero_share` is the share of conflict-level draws set to zero; draws kept equal \(200 \times (1 - \texttt{cluster\_zero\_share})\) |
| `*.png` | LASSO paths, baseline results, and forest plots written beside the scripts |

### 2. Descriptive statistics (`modeling_on_full_data/descriptive_stats`)

| File | Description |
| :--- | :--- |
| `desc_stats.py` | Summary statistics and distribution plots. Reads `../pad_merged.csv` |
| `*.png` | Plots for continuous, categorical, and binary predictors |

### 3. Provision table (`modeling_on_full_data/provision_table`)

| File | Description |
| :--- | :--- |
| `produce_prov_table.py` | Formatted table of PAD provisions |
| `pad_provisions_table.png` | Rendered table |

### 4. PAD-only analysis (`modeling_pad_alone`)

The same Kaplan-Meier, Cox, and LASSO pipeline on PAD provisions alone (52 predictors).

| File | Description |
| :--- | :--- |
| `PAD_final.xlsx` | Analysis file derived from UCDP PAD |
| `kaplan_meier.py` | Stratified Kaplan-Meier curves and log-rank tests |
| `baseline_cox.py` | Unpenalized Cox model with clustered standard errors |
| `cox_lasso.py` | LASSO-penalized Cox model and bootstrap intervals |

---

## Model Specification

`modeling_on_full_data` uses a 67-column design matrix:

* **28** binary peace-agreement provisions
* **5** external binary context variables (prior rebel victory, neighbor at war, and related flags)
* **1** PAD ordinal variable (years since last conflict)
* **10** continuous external covariates, standardized before estimation
* **10** dummy variables (region, incompatibility, agreement type, and frame)
* **13** interaction terms

`modeling_pad_alone` uses the same provision, dummy, and interaction blocks without the external covariates (52 columns).

**Methodological notes:**

* **Interactions.** Products of standardized main effects, then residualized on those main effects and an intercept.
* **Clustering.** Agreements are clustered by conflict. Unpenalized Cox standard errors use the Lin-Wei sandwich estimator. A missing `conflict_id` is its own cluster.
* **Bootstrap.** LASSO intervals use 200 resamples. One resample draws agreements and another draws whole conflicts. Failed fits are omitted. A draw shrunk to exactly zero is counted separately, because treating it as an HR of 1 pins a percentile endpoint on 1. That endpoint means the term was dropped in at least 2.5% of draws. The number of conflict-level draws that keep a term is \(200 \times (1 - \texttt{cluster\_zero\_share})\).
* **Durations.** Nonpositive durations are dropped. A `termdur` code of −99 is recoded to 0.
* **Proportional hazards.** The baseline model tests the assumption with Schoenfeld residuals. Violators can be re-entered as time-varying terms through a person-period expansion in `statsmodels`. The violator list in the LASSO scripts starts empty. `robustness_checks.py` reports the Grambsch–Therneau flags and a national-dialogue \(\log(t)\) term, and does not add time-varying terms to the reported fit.

---

## Data Sources

The agreements come from the [UCDP Peace Agreement Dataset (PAD) version 22.1](https://ucdp.uu.se/downloads/#peaceagreement). Variable definitions follow the PAD codebook. External covariates were joined from V-Dem, conflict history, and macroeconomic series. Interstate agreements in the robustness frame are flagged from UCDP/PRIO Armed Conflict Dataset v22.1 (`type_of_conflict == 2`).

---

## Requirements

Install the pinned dependencies with:

```bash
pip install -r requirements.txt
```

The LASSO fits use scikit-survival 0.23.1. The pins were installed on Python 3.12.

## Usage

Scripts locate their data from their own file path, so they can be run from any working directory. Console output and figures are written next to the script that produced them.

```bash
python modeling_on_full_data/cox_lasso_full.py
python modeling_on_full_data/robustness_checks.py
python modeling_on_full_data/paper_figures.py
```

`paper_figures.py` writes `KaplanMeierByRegion.pdf` and `VARIABLEIMPORTANCE.pdf` to `papers/`.
