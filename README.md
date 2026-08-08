
<p align="center">
• <a href="https://mayzune.com/"><strong>May Zune</strong></a> •
<a href="https://github.com/hellomayzune"><strong>GitHub</strong></a> •
<a href="https://orcid.org/0000-0003-0282-2633"><strong>ORCID</strong></a> •
<a href="https://scholar.google.com/citations?user=LmP8B_4AAAAJ&hl=en"><strong>Google Scholar</strong></a> •
<a href="https://www.researchgate.net/profile/May-Zune"><strong>ResearchGate</strong></a> •
<a href="https://www.linkedin.com/in/mayzune//"><strong>Linkedin</strong></a> •
</p>

<p align="center">
  <a href="https://www.imperial.ac.uk/">
    <img src="https://img.shields.io/badge/Imperial%20College%20London-000025?style=flat" alt="Imperial College London">
  </a>
  <a href="https://github.com/hellomayzune/Imperial-College-Capstone-Black-box-Optimisation/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white" alt="Python Version">
  </a>
  <a href="https://scikit-learn.org/stable/">
    <img src="https://img.shields.io/badge/scikit--learn-v1.4.0-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  </a>
</p>


# EPC-Based Energy Consumption Prediction

A collection of practical machine learning notebooks developed as part of my learning through the **Imperial College London Professional Certificate in Machine Learning / AI**.

The notebooks demonstrate how different ML methods can be applied to **EPC-based energy consumption prediction**, using an Energy Performance Certificate (EPC) dataset for **Sheffield** (`EPC_Sheffield_20260414-clean.csv`, ~209,000 records).

The primary purpose of this repository is **technical learning and demonstration** — exploring machine learning concepts, understanding their practical applications, comparing algorithms, and evaluating their limitations, rather than producing a production-ready model.

## Contents

| Notebook | Task type | Model | Headline result |
|---|---|---|---|
| `EPC-Based_Energy_Consumption_Prediction_Using_Logistic_Regression.ipynb` | Binary classification (high vs. low energy consumption) | Logistic Regression | Accuracy 0.934, ROC AUC 0.983 |
| `EPC-Based_Energy_Consumption_Prediction_Using_KNN_Regression.ipynb` | Regression | K-Nearest Neighbours (k=5) | R² 0.878, MAE 24.51, RMSE 38.72 |
| `EPC-Based_Energy_Consumption_Prediction_Using_SVR.ipynb` | Regression | Support Vector Regression (RBF kernel) | R² 0.911, MAE 15.26, RMSE 34.07 |
| `EPC-Based_Energy_Consumption_Prediction_Using_Random_Forest.ipynb` | Regression | Random Forest Regressor | R² 0.907 |
| `EPC-Based_Energy_Consumption_Prediction_Using_Bayesian_Optimisation.ipynb` | Regression + hyperparameter optimisation | HistGradientBoostingRegressor, tuned via custom Bayesian Optimisation (Gaussian Process surrogate) | R² 0.941, MAE 13.52, RMSE 27.04 |

## Dataset

All notebooks use the same source data:

- **Dataset:** Sheffield EPC dataset, ~209,000 rows.
- **Target variable:** `energy_consumption_current` (used directly for regression; thresholded at the median for the Logistic Regression classification task).
- **Dropped columns:** `certificate_number`, `postcode`, `constituency_label`, `constituency` (identifiers/location fields not used as predictors).
- **Remaining features:** property characteristics such as floor area, current energy efficiency, heating systems, building age, and number of rooms.

## Shared approach across notebooks

Despite covering five different algorithms, the notebooks follow a consistent structure and preprocessing approach so that results are broadly comparable:

1. **Load & clean** — read the CSV and drop the four identifier/location columns.
2. **Define target/features** — `energy_consumption_current` as the (regression or thresholded classification) target; all remaining columns as features.
3. **Preprocessing pipeline** (via `ColumnTransformer` + `Pipeline`):
   - Numerical features: median imputation + `StandardScaler`.
   - Categorical features: most-frequent imputation + `OneHotEncoder`.
4. **Train/test split** — an 80/20 split (stratified for the classification task), with `random_state=42` for reproducibility.
5. **Evaluation** — RMSE, MAE, and R² for regression tasks; accuracy, ROC AUC, confusion matrix, and classification report for the classification task.
6. **Diagnostics & visualisation** — actual vs. predicted plots, residual plots/distributions, and (where applicable) feature importance or coefficient analysis.
7. **Notebook structure** — each notebook ends with a **Limitations** section and an **Applicability to Real-World Studies** section, framing results honestly as a learning exercise rather than a deployment-ready model.

## Where the methods differ

| Aspect | Logistic Regression | KNN | SVR | Random Forest | Bayesian Optimisation |
|---|---|---|---|---|---|
| Problem framing | Classification (high/low, split at median) | Regression | Regression | Regression | Regression |
| Sample size used | Full dataset (stratified split) | Subsampled (30k train / 5k test) | Subsampled (10k rows) | Full dataset | Subsampled (20k rows) |
| Hyperparameter tuning | None (default settings) | Manual sensitivity analysis over a small range of `k` | None (manually chosen `C=100`, `epsilon=0.1`) | Fixed settings (`n_estimators=100`, `max_depth=20`) | Custom Bayesian Optimisation loop (Gaussian Process surrogate, Expected Improvement acquisition) over 4 hyperparameters |
| Interpretability | Coefficients (feature-level direction/strength) | None built in | None built in | Feature importances | None beyond the underlying HistGradientBoosting model |
| Validation strategy | Single stratified split | Single split | Single split | Single split | 3–5-fold cross-validation used *during* optimisation; single split for final evaluation |
| Distinctive focus | Classification framing of a regression target; ROC/PR analysis | Effect of `k` on the bias–variance trade-off | Kernel-based non-linear regression; permutation importance & Q-Q residual diagnostics | Ensemble tree-based regression; feature importance ranking | Systematic hyperparameter search as an alternative to manual/grid tuning |

## Key observations

- **Bayesian-optimised HistGradientBoosting achieved the best regression performance** (R² 0.941), suggesting that systematic hyperparameter search offers a meaningful improvement over the fixed/manually-chosen settings used in the other regression notebooks — though it also uses a smaller sample and a limited search budget.
- **SVR and Random Forest performed similarly** (R² ~0.91 and ~0.907 respectively) despite being very different model families (kernel-based vs. ensemble tree-based).
- **KNN performed the weakest** of the regressors tested (R² 0.878), and is also the most computationally expensive at prediction time, especially after one-hot encoding expands the feature space.
- **Logistic Regression performed strongly as a classifier** (accuracy 0.934, ROC AUC 0.983) when the problem is simplified to a binary high/low split — but this discards information about the actual magnitude of energy consumption.
- Across the tree-based models, **`current_energy_efficiency` consistently emerges as the most important feature**, which is flagged in the Random Forest notebook as a potential target-leakage concern worth further investigation, since it is closely related to the energy consumption target itself.

## Limitations (repository-wide)

- Most notebooks use a single train/test split rather than repeated cross-validation for final reported metrics.
- Several notebooks subsample the ~209k-row dataset (10k–30k rows) for computational efficiency, which may affect generalisability.
- Hyperparameter tuning is inconsistent across notebooks — only the Bayesian Optimisation notebook performs systematic tuning.
- No single notebook directly benchmarks all five methods against each other on identical train/test splits; the comparison above is based on each notebook's own reported results.
- Results are specific to the Sheffield EPC dataset and have not been validated on other regions, time periods, or property types.

## Applicability

These notebooks are intended as **learning and demonstration artefacts**: they illustrate how classification and regression methods, preprocessing pipelines, and hyperparameter optimisation can be applied to a real-world, moderately messy tabular dataset (EPC records). With further validation — larger samples, consistent cross-validation, and systematic tuning across all models — similar pipelines could support real-world applications such as energy-efficiency assessment, targeting properties for retrofit interventions, and housing policy analysis.

