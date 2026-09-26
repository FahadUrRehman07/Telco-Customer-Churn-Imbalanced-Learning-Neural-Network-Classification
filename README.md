# Telco Customer Churn — Imbalanced Learning & Neural Network Classification

A reproducible machine-learning project for predicting customer churn while studying how class imbalance and decision thresholds affect minority-class detection.

## Project Status

**Day 5 of 10 — Random under-sampling comparison**

The project now includes a leakage-safe random under-sampling experiment. Only the training split is rebalanced; the stratified test split remains untouched for a fair comparison with the original ANN baseline.

## Workflow

1. Project structure and cleanup — complete
2. Leakage-safe preprocessing pipeline — complete
3. Baseline ANN + evaluation — complete
4. Class-imbalance analysis — complete
5. Random under-sampling comparison — complete
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

## Imbalance analysis

`src/imbalance.py` reports:

- Total rows
- Positive and negative churn counts
- Positive and negative class rates
- Minority-to-majority ratio
- Majority-class accuracy baseline

## Random under-sampling

`src/sampling.py` adds:

- Majority-class down-sampling to the minority-class size
- Shuffling after resampling
- Reuse of the same leakage-safe ANN pipeline
- Evaluation on the original untouched test split
- Sample-count reporting so the training-distribution change is explicit

The helper raises an error for non-binary targets instead of silently applying an invalid strategy.

## Usage

Import `load_telco_csv`, `train_baseline`, `analyze_target_distribution`, and `train_undersampled` from `src`. Keep the raw dataset outside Git unless its redistribution terms permit committing it.

## Repository structure

README.md
requirements.txt
src/
  __init__.py
  baseline.py
  imbalance.py
  preprocessing.py
  sampling.py
