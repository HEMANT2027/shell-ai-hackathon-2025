# Shell.ai Hackathon 2025: Fuel Blend AI Challenge

A comprehensive solution for the Fuel Blend AI Challenge, detailing the end-to-end process from data analysis and feature engineering to model experimentation and final ensemble strategy. All work is contained within the main Jupyter Notebook.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📝 Table of Contents
- [Problem Statement](#-problem-statement)
- [Evaluation Metric](#-evaluation-metric)
- [Methodology](#-methodology)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Modeling Experiments](#3-modeling-experiments)
- [Final Approach & Results](#-final-approach--results)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [⚖License](#️-license)

---

## Problem Statement

The challenge is to develop a model that accurately estimates 10 distinct properties of a blended fuel. The input data consists of the fractional amounts of 5 different components and 10 known properties for each of those components. The goal is to predict the final 10 properties of the resulting mixture.

---

## Evaluation Metric

The evaluation metric for this competition is the **Mean Absolute Percentage Error (MAPE)**, which is calculated using the formula:

$$
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{A_i - P_i}{A_i} \right| \times 100\%
$$

Where:
- $n$ is the number of data points.
- $A_i$ is the actual value.
- $P_i$ is the predicted value.

The final leaderboard score is calculated from the MAPE as:

$$
\text{Score} = \max(0, 100 - \text{MAPE})
$$

---

## 🛠️ Methodology

### 1. Exploratory Data Analysis (EDA)

Initial analysis was performed to understand the relationships within the data:
* **Correlation Heatmap**: A heatmap visualizing the correlations between all 55 input features and the 10 target blend properties was generated. This helped identify initial linear relationships, with a notable strong positive correlation between `BlendProperty1` and `BlendProperty7`.
* **Distribution Analysis**: Boxplots were created for each of the 10 `BlendProperty` targets to inspect their distributions, central tendencies, and the presence of outliers.

### 2. Feature Engineering

A key part of this project was creating new features to capture the complex, non-linear interactions between fuel components. The following features were engineered:

* **General Features**:
    1.  **Weighted Averages**: Overall property contributions based on component fractions.
    2.  **Property Statistics**: The min, max, mean, and standard deviation for each of the 10 properties across the 5 components.
    3.  **Interaction Features**: The product of each component's fraction and its properties (`fraction × property`).
    4.  **Dominant Component**: An identifier for the component with the largest fraction in the blend.

* **Synergistic & Antagonistic Features**:
    1.  **Pairwise Interactions**: Products of component fractions (`fraction_i × fraction_j`) to model the effect of two components being present simultaneously.
    2.  **Deviation from Linearity**: Features measuring how much a property deviates from a simple weighted average. This is a direct attempt to model synergy and includes deviations for the dominant component's property, the max property, and the min property.
    3.  **Heterogeneity Interactions**: A feature combining composition diversity (calculated via entropy) with property diversity (standard deviation), designed to capture complex interactions when diverse components are mixed.

The full implementation can be found in the `add_engineered_features` function in the notebook.

### 3. Modeling Experiments

A wide range of models and feature selection techniques were tested to find the best-performing approach. The leaderboard scores for each experiment are noted below:

| Experiment                                     | Score (100 - MAPE) |
| ---------------------------------------------- | :----------------: |
| KFold CV on CatBoost                           |      74.53801      |
| CatBoost with SelectKBest Feature Selection    |      49.76943      |
| Random Forest with RFE                         |      75.81276      |
| KFold CV on Linear Regression                  |       ~10.0        |
| Linear Regression with SelectKBest             |      44.62569      |
| Random Forest after PCA                        |      23.44412      |
| LightGBM Model                                 |      79.47185      |
| LightGBM with Feature Importance Selection     |      67.31835      |
| Tuned LightGBM Model                           |      74.27700      |
| MultiOutput Lasso with RFE                     |      62.08373      |
| **MultiOutput ElasticNet & Ridge with RFE** |    **~35.86159** |
| BaggingRegressor with XGBoost (KFold CV)       |      74.70199      |
| Bagging with Ridge Regression                  |      57.26071      |

---

## Final Approach & Results

The best results were not achieved by a single model but by **ensembling the predictions of the top-performing models**. The final strategy involved combining two strong submissions:
1.  **`df_best` (Score: 88.57817)**: A hybrid model that used different regressors for different blend properties.
2.  **`df_cat` (Score: 82.24153)**: A submission generated using a well-tuned CatBoost Regressor.

Two ensembling techniques were used to generate the final submissions:

1.  **Weighted Blending**: The two prediction files were blended using a variety of weights (e.g., 95% `df_best` + 5% `df_cat`, 90% `df_best` + 10% `df_cat`, etc.).
2.  **Cherry-Picking**: The best submission (`df_best`) was used as a base, and its predictions for one target column at a time were replaced with the corresponding predictions from the CatBoost submission (`df_cat`).

This ensemble approach leverages the strengths of different models, significantly improving the final score over any single model.

---

## Repository Structure

Since all work was performed sequentially, the entire project is contained within a single Jupyter Notebook for easy reproducibility.
