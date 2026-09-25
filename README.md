# Telco Customer Churn — Imbalanced Learning & Neural Network Classification

A reproducible machine-learning project for predicting customer churn while studying how class imbalance and decision thresholds affect minority-class detection.

## Project Status

**Day 4 of 10 — Class-imbalance analysis**

The project now includes a target-distribution diagnostic before any resampling strategy is introduced. This makes the later under-sampling, over-sampling, and SMOTE comparisons easier to interpret.

## Workflow

1. Project structure and cleanup — complete
2. Leakage-safe preprocessing pipeline — complete
3. Baseline ANN + evaluation — complete
4. Class-imbalance analysis — complete
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

## Imbalance analysis

src/imbalance.py reports:

- Total rows
- Positive and negative churn counts
- Positive and negative class rates
- Minority-to-majority ratio
- Majority-class accuracy baseline

The diagnostic is intentionally model-free. It quantifies the problem before changing the training distribution.

## Usage

Import load_telco_csv, train_baseline, and analyze_target_distribution from src. Keep the raw dataset outside Git unless its redistribution terms permit committing it.

## Repository structure

README.md
requirements.txt
src/
  __init__.py
  baseline.py
  imbalance.py
  preprocessing.py
