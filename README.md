# Soccer Player Market Value Prediction

### Predicting Soccer Player Valuations Through Machine Learning

**Authors:** Gurpreet Singh and MD Rafi  
**Course:** CISC-5790 - Data Mining  
**Institution:** Fordham University

## Overview

This project explores how machine learning can estimate soccer players' market values using player attributes, skill ratings, and financial information. Using a dataset of approximately **17,000 players and 51 features**, the study compares regression models, tree-based ensembles, and neural networks to investigate the factors associated with player valuation.

The project focuses on four main tasks:

1. **Data Preparation and Exploration** - examine distributions, handle missing values, and investigate relationships between player attributes and market value.
2. **Feature Engineering** - construct performance and development indicators to complement the original player attributes.
3. **Market Value Prediction** - train and compare regression models using a held-out test set and cross-validation.
4. **Feature Importance Analysis** - identify influential attributes using Random Forest importance and Recursive Feature Elimination (RFE).

## Data

The dataset is **Football Players Data**, published on Kaggle by Masood Ahmed and collected from **SoFIFA** between **October 18 and October 31, 2023**.

**Dataset:** https://www.kaggle.com/datasets/maso0dahmed/football-players-data

| Category | Examples |
| --- | --- |
| Player characteristics | Age, height, weight |
| Player ratings | Overall rating, potential |
| Technical attributes | Finishing, dribbling, positioning, ball control |
| Defensive attributes | Marking, standing tackle, sliding tackle |
| Financial information | Wages and market value |

**Prediction target:** `value_euro`, representing the dataset's player valuation in euros. Experiments also use a log-transformed target, `value_euro_log`.

The modeling approach focuses on numerical predictors. These valuations and ratings come from the SoFIFA dataset and should be distinguished from observed transfer fees.

## Methods and Tools

- **Language:** Python
- **Classical machine learning:** scikit-learn
- **Deep learning:** TensorFlow / Keras
- **Visualization:** Matplotlib and Seaborn
- **Regression:** Linear Regression, Ridge, Lasso, and a custom closed-form Linear Regression implementation
- **Tree-based models:** Random Forest and Gradient Boosting
- **Neural networks:** MLP Regressor and a Keras sequential network
- **Ensembles:** Stacking and Voting Regressors
- **Interpretation:** Random Forest feature importance, correlation analysis, and RFE

## Project Pipeline

```text
Data Collection
      |
Data Cleaning and Preprocessing
      |
Exploratory Data Analysis
      |
Feature Engineering
      |
Training / Test Split
      |
Model Training and Hyperparameter Search
      |
Evaluation and Model Comparison
      |
Feature Importance and Findings
```

## Data Preparation and Feature Engineering

The paper describes the following preparation steps:

- Remove records missing critical target information.
- Exclude sparsely populated or unused categorical fields from numerical modeling.
- Standardize numerical inputs using z-score normalization where applicable.
- Inspect extreme values and retain high-value observations in the reported experiments.
- Log-transform market value to reduce its right-skewed distribution.

Two additional predictors capture combinations of existing player attributes:

| Engineered predictor | Calculation | Purpose |
| --- | --- | --- |
| `performance_index` | Mean of overall rating, potential, and composure | Summarize player quality using a composite measure |
| `age_to_potential_ratio` | Age divided by potential | Represent age relative to the player's potential rating |

The transformed target, `value_euro_log`, is a target representation rather than an independent player attribute.

## Exploratory Data Analysis

The analysis uses histograms, boxplots, scatterplots, and a correlation heatmap to examine:

- The skewed distribution of player market values.
- Relationships between valuation, overall rating, potential, and wages.
- Correlations among numerical predictors.
- Extreme observations and potential multicollinearity.

## Modeling and Evaluation

The study reports a **66% training / 34% test split** and **5-fold cross-validation**. Models are evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of determination

Grid searches explore Random Forest parameters and MLP configurations. Ridge and Lasso are evaluated across regularization strengths. Stacking and Voting Regressors combine predictions from multiple base models.

### Results Reported in the Paper

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.18 | 0.23 | 0.97 |
| Ridge Regression | 0.169702 | 0.226939 | 0.973164 |
| Lasso Regression | 0.169714 | 0.227218 | 0.973098 |
| **Random Forest** | **0.05** | **0.09** | **1.00** |
| Gradient Boosting | 0.08 | 0.10 | 0.99 |
| MLP Regressor - initial configuration | 0.25 | 0.31 | 0.95 |
| MLP Regressor - tuned configuration | 2.15 | 3.06 | -3.72 |

Additional experiments report:

| Approach | Reported MAE |
| --- | ---: |
| Stacking Regressor | 0.05 |
| Voting Regressor | 0.09 |
| Keras sequential network | 0.1574 |

**Interpretation:** Random Forest achieved the strongest standalone results in the paper, while stacking matched its reported MAE. The tuned MLP performed worse than its initial configuration, illustrating that hyperparameter search does not necessarily improve held-out performance.

**Metric context:** These are the paper's reported values, not independently reproduced results. The study describes a log-transformed target, but the metric tables do not consistently specify the evaluation scale; MAE and RMSE should therefore not be interpreted as euro amounts or assumed directly comparable across every experiment. The reported R² of 1.00 is rounded and does not establish error-free prediction. Confirming the target scale, preprocessing order, and exclusion of target-derived inputs requires reviewing the notebook.

### Random Forest Configuration

The paper reports the following selected parameters:

| Parameter | Value |
| --- | --- |
| `n_estimators` | 200 |
| `max_depth` | 20 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `max_features` | `None` |

The reported cross-validated RMSE is **0.10**, compared with a test RMSE of **0.09**.

## Feature Importance

Random Forest ranks the following attributes as the ten most influential predictors:

| Rank | Attribute |
| ---: | --- |
| 1 | Overall Rating |
| 2 | Age-to-Potential Ratio |
| 3 | Potential |
| 4 | Finishing |
| 5 | Performance Index |
| 6 | Age |
| 7 | Positioning |
| 8 | Volleys |
| 9 | Sliding Tackle |
| 10 | Long Shots |

RFE also identifies age, overall rating, potential, finishing, dribbling, and the engineered predictors among the important attributes. These findings describe predictive associations within this dataset rather than causal effects on transfer prices.

## Limitations and Future Work

- **Time coverage:** The dataset is an October 2023 snapshot and does not capture changes in valuation over time.
- **Data context:** SoFIFA ratings and valuations may reflect relationships specific to their source; performance on actual transfer fees requires separate evaluation.
- **Feature coverage:** Numerical modeling excludes categorical context such as club affiliation.
- **Position-specific models:** Separate models for goalkeepers, defenders, midfielders, and forwards could capture different valuation patterns.
- **Reproducibility:** Notebook verification should confirm target handling, train-only preprocessing, and evaluation on the original euro scale.
- **Future evaluation:** Updated data and temporal holdouts would help assess performance beyond the original snapshot.

## Technical Report

The accompanying report, **Predicting Soccer Player's Market Value**, documents the methodology, model comparisons, visualizations, and findings. Upload `Soccer_Player_Market_Value_Report.pdf` alongside this README to include it in the repository.

## Running the Project

The notebook and dependency files were not provided with the report used to prepare this README. Exact installation steps, dataset paths, and execution instructions should be added after the project notebook is included and verified.

## Acknowledgments

- **Gurpreet Singh and MD Rafi** - project authors
- **Fordham University, CISC-5790 Data Mining** - academic context
- **Masood Ahmed / Kaggle** - dataset publication
- **SoFIFA** - underlying player data source
