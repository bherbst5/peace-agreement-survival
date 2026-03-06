# Peace Agreement Duration — Survival Analysis

Code for: *[Toward Durable Peacemaking: Evaluating Intrastate Peace Agreement Provisions via Penalized Survival Analysis]*)

---

## Overview

This repository contains all analysis code and output figures for a survival
analysis of peace agreement duration using the Peace Agreements Database (PAD).
The outcome of interest is time to agreement collapse (failure), with agreements
still in force at the end of the observation window (31 December 2021) treated
as right-censored.

Three models are implemented and compared:
- Kaplan-Meier non-parametric survival curves (stratified by key covariates)
- Baseline Cox Proportional Hazards model with cluster-robust standard errors
- LASSO-penalized Cox Proportional Hazards model with cross-validated penalty selection

---

## Repository Contents

| File | Description |
|------|-------------|
| `PAD_final.xlsx` | Analysis dataset derived from the UCDP Peace Agreements Dataset |
| `kaplan_meier.py` | Kaplan-Meier survival curves, log-rank tests |
| `kaplan_meier_curves.png` | Output figures from `kaplan_meier.py` |
| `baseline_cox.py` | Unpenalized Cox PH model with cluster-robust SEs (Lin-Wei sandwich), Grambsch-Therneau PH test, and extended model for PH violators |
| `baseline_results.png` | Output figures from `baseline_cox.py` |
| `cox_lasso.py` | LASSO-penalized Cox PH via scikit-survival, 5-fold CV, bootstrap CIs |
| `cox_lasso_results.png` | Output figures from `cox_lasso.py` |

---

## Data

The dataset (`PAD_final.xlsx`) is derived from the
[UCDP Peace Agreement Dataset (PAD) version 22.1] (https://ucdp.uu.se/downloads/#peaceagreement),
maintained by the Uppsala Conflict Data Program.

> The dataset is included in this repository. Variable definitions follow the
> PAD codebook, available at https://ucdp.uu.se/downloads/peace/ucdp-codebook-peace-agreements-221.pdf

---

## Model Specification

All three scripts share the same design matrix (52 columns):
- 28 binary provision indicators
- 1 ordinal variable (`termdur`)
- 10 dummy-encoded categorical variables (region, incompatibility type,
  agreement type, frame; reference categories: Africa, Territory, Full, Frame 1)
- 13 interaction terms, residualized against their constituent main effects
  to ensure orthogonality

Agreements are clustered within conflicts; all Cox model standard errors
are cluster-robust (Lin-Wei 1989 sandwich estimator, clustered by `conflict_id`).

---

## Requirements

Python 3.8 or higher. Install dependencies with:
```bash
pip install scikit-survival pandas numpy matplotlib openpyxl scipy scikit-learn
```

Tested with scikit-survival 0.23.1.

---

## Usage

Run each script from the repository root with the dataset in the same directory:
```bash
python kaplan_meier.py
python baseline_cox.py
python cox_lasso.py
```

Each script saves its output figure to the working directory and prints
a results summary to the console.