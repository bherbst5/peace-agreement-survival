# Peace Agreement Duration — Survival Analysis

Code for: *[Toward Durable Peace Agreements: Provision Selection and Contextual Risk in Intrastate Conflict Resolution]*

---

## Overview

This repository contains all analysis code, data processing scripts, and output figures for a survival analysis of peace agreement duration. The outcome of interest is time to agreement collapse (failure), with agreements still in force at the end of the observation window (December 31, 2021) treated as right-censored.

The project has evolved to analyze both the foundational provisions of the agreements and the broader macro-political/economic contexts in which they were signed. As such, the repository is divided into models run exclusively on the Peace Agreement Dataset (PAD) and models run on a fully merged dataset containing external covariates (e.g., V-Dem indices, conflict history, and macroeconomic indicators).

Three primary analytical approaches are implemented:
* Kaplan-Meier non-parametric survival curves (stratified by key covariates)
* Baseline Cox Proportional Hazards models (with cluster-robust standard errors and extended time-varying terms for proportional hazards violators)
* LASSO-penalized Cox Proportional Hazards models (featuring 5-fold cross-validation with the Verweij–Van Houwelingen correction and bootstrap confidence intervals)

---

## Repository Structure

### 1. Primary Analysis (`/modeling_on_full_data`)
This directory contains the main analysis utilizing the expanded 67-predictor dataset.

| File | Description |
| :--- | :--- |
| `pad_merged.csv` | Primary dataset (PAD merged with external macroeconomic and conflict variables) |
| `kaplan_meier_full.py` | Generates stratified Kaplan-Meier curves and log-rank tests |
| `baseline_cox_full.py` | Unpenalized Cox PH model, PH assumption testing, and extended time-varying model |
| `cox_lasso_full.py` | LASSO-penalized Cox PH via scikit-survival, featuring variable selection |
| `*.png` files | Output visualizations (LASSO paths, baseline results, univariate forest plots) |

### 2. Exploratory Data Analysis (`/modeling_on_full_data/descriptive_stats`)
Scripts and outputs for visualizing the distributions of the merged dataset.

| File | Description |
| :--- | :--- |
| `desc_stats.py` | Generates summary statistics and distribution visualizations |
| `*.png` files | Output plots for continuous, categorical, and binary predictors |

### 3. Data Summaries (`/modeling_on_full_data/provision_table`)
| File | Description |
| :--- | :--- |
| `produce_prov_table.py` | Script to generate a formatted summary table of PAD provisions |
| `pad_provisions_table.png` | Output visualization of the provision frequencies |

### 4. Legacy / Base PAD Analysis (`/modeling_pad_alone`)
This directory contains the initial modeling pipeline run exclusively on the base PAD data, prior to the inclusion of external covariates.

| File | Description |
| :--- | :--- |
| `PAD_final.xlsx` | Base dataset derived directly from UCDP PAD |
| `*.py` files | Base scripts for Kaplan-Meier, Cox PH, and LASSO models |

---

## Model Specification

The primary scripts in the `modeling_on_full_data` directory share a comprehensive 67-column design matrix:
* **28** binary peace agreement provision indicators
* **5** external binary context variables (e.g., prior rebel victory, neighbor at war)
* **1** continuous variable from PAD (term duration)
* **10** continuous external covariates (standardized prior to estimation)
* **10** dummy-encoded categorical variables (region, incompatibility type, agreement type, frame)
* **13** theoretically-motivated interaction terms

**Methodological Notes:**
* **Interactions:** Constructed as products of standardized main effects, then orthogonalized against their constituent predictors to prevent lower-order confounding.
* **Clustering:** Agreements are clustered within conflicts. All unpenalized Cox model standard errors are cluster-robust (Lin-Wei sandwich estimator).
* **Time-Varying Covariates:** The baseline model evaluates proportional hazards (PH) assumptions via Schoenfeld residuals. Violators are re-entered as time-varying terms using a person-period expansion via `statsmodels`.

---

## Data Sources

The foundational dataset is derived from the [UCDP Peace Agreement Dataset (PAD) version 22.1](https://ucdp.uu.se/downloads/#peaceagreement), maintained by the Uppsala Conflict Data Program.

Variable definitions follow the official PAD codebook. External covariates were joined from standard international relations datasets (e.g., V-Dem, World Bank).

---

## Requirements

Python 3.8 or higher is required. Install the necessary dependencies using `pip`:

```bash
pip install scikit-survival pandas numpy matplotlib openpyxl scipy scikit-learn statsmodels
```

*Note: Tested with scikit-survival 0.23.1.*

## Usage

To reproduce the analysis, run the scripts from within their respective directories. For example, to run the primary LASSO analysis:

```bash
cd modeling_on_full_data
python cox_lasso_full.py
```

Each script will output diagnostic information to the console and save the corresponding high-resolution figures directly to the working directory.