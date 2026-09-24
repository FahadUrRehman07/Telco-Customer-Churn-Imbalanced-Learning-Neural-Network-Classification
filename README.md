# Telco Customer Churn — Imbalanced Learning & Neural Network Classification

A reproducible machine-learning project for predicting customer churn while studying how class imbalance and decision thresholds affect minority-class detection.

## Project Status

**Day 3 of 10 — Baseline ANN and evaluation**

The project now has a leakage-safe preprocessing pipeline and a reproducible neural-network baseline. Subsequent experiments will change one imbalance strategy at a time so their effect can be compared fairly.

## Workflow

1. Project structure and cleanup — complete
2. Leakage-safe preprocessing pipeline — complete
3. Baseline ANN + evaluation — complete
4. Class-imbalance analysis
5. Random under-sampling comparison
6. Over-sampling comparison
7. SMOTE comparison
8. Threshold and metric analysis
9. Reproducibility and code cleanup
10. Final documentation and completion checks

## Baseline design

- Stratified 80/20 train-test split
- Median imputation and standardization for numeric features
- Most-frequent imputation and one-hot encoding for categorical features
- MLPClassifier with two hidden layers: 64 → 32
- Fixed random_state=42
- Early stopping using only the training split
- Accuracy, precision, recall, F1-score, and ROC-AUC

The test set remains untouched until final evaluation.

## Usage

Import `load_telco_csv` and `train_baseline` from `src.baseline`, then pass the loaded Telco dataframe to `train_baseline`. Keep the raw dataset outside Git unless its redistribution terms permit committing it.

## Repository structure

README.md
requirements.txt
src/
  __init__.py
  baseline.py
  preprocessing.py