# Purchasing Intention Prediction: Supervised Classification

Predicts whether an online shopping session ends in a purchase by comparing three classifiers in a scikit-learn pipeline and selecting the best with cross-validation.

## Overview

The data is the UCI *Online Shoppers Purchasing Intention* dataset: 12,330 sessions, each from a different user over one year. The goal is to flag likely buyers and likely abandoners so that promotions can be targeted.

## Approach

1. Stratified 80/20 train and test split (`random_state=21`) to preserve the class ratio.
2. A `Pipeline` per model: `StandardScaler` followed by the classifier, so scaling is fit only on training folds.
3. Three classifiers compared: **Logistic Regression**, **K-Nearest Neighbors** and **Decision Tree**.
4. `GridSearchCV` with 5-fold shuffled cross-validation (scoring: accuracy) tunes each model's hyperparameters.
5. The best model is evaluated once on the held-out test set with accuracy, precision, recall, F1, a classification report and a confusion matrix.

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python, Jupyter Notebook |
| ML | scikit-learn (Pipeline, GridSearchCV, KFold) |
| Data and plots | pandas, seaborn, matplotlib |

## Results

Best model: **Decision Tree** (`max_depth=5`, `min_samples_split=2`), cross-validated accuracy 0.899.

Held-out test set (2,466 sessions: 2,084 no purchase, 382 purchase):

| Metric (purchase class) | Value |
| --- | --- |
| Accuracy | 0.897 |
| Precision | 0.727 |
| Recall | 0.537 |
| F1 | 0.618 |

Always predicting "no purchase" would already reach about 84.5% accuracy, so precision, recall and F1 for the purchase class are more informative than accuracy here.

## Run it

1. Download the data from the [UCI repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). The notebook reads a local copy named `online_shoppers_intention - Copy.csv` in which the `Month` and `VisitorType` columns were converted to integers.

2. Install dependencies and run:

   ```bash
   pip install pandas scikit-learn seaborn matplotlib jupyter
   jupyter notebook OSPI_Pdn_Rel_1.ipynb
   ```

## Known limitations

- The dataset is imbalanced and model selection used accuracy, which favors the majority class.
- `Month` and `VisitorType` are integer-encoded; for linear and distance-based models one-hot encoding is more appropriate.
- The preprocessed CSV is not included in this repository.

## Next steps

- Select models on F1 or PR-AUC, with class weights and decision-threshold tuning based on business cost.
- One-hot encode categorical columns inside the pipeline.
- Compare tree ensembles (random forest, gradient boosting) and report feature importances.

## Skills demonstrated

Supervised learning, pipeline design, cross-validation, hyperparameter tuning, model evaluation on imbalanced data.
