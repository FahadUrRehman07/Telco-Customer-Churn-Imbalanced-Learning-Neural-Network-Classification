# Telco Customer Churn — Imbalanced Learning & Neural Network Classification

A reproducible machine-learning project for predicting customer churn while studying how class imbalance and decision thresholds affect minority-class detection.

## Project Status

**Day 7 of 10 — SMOTE comparison**

The project now includes a leakage-safe SMOTE experiment. Synthetic samples are generated only after fitting preprocessing on the training split; the stratified test split remains untouched for a fair comparison with earlier experiments.

## Workflow

1. Project structure and cleanup — complete
2. Leakage-safe preprocessing pipeline — complete
3. Baseline ANN + evaluation — complete
4. Class-imbalance analysis — complete
5. Random under-sampling comparison — complete
6. Random over-sampling comparison — complete
7. SMOTE comparison — complete
8. Threshold and metric analysis
9. Reproducibility and code cleanup
10. Final documentation and completion checks

## Sampling comparisons

- `src/sampling.py`: random under-sampling.
- `src/oversampling.py`: random over-sampling.
- `src/smote.py`: synthetic minority over-sampling after leakage-safe encoding.
- All experiments keep the test split untouched and report sample counts plus accuracy, precision, recall, F1-score, and ROC-AUC.

## Usage

Import `train_baseline`, `train_undersampled`, `train_oversampled`, and `train_smote` from `src`. Keep the raw dataset outside Git unless its redistribution terms permit committing it.

## Repository structure

README.md
requirements.txt
src/
  __init__.py
  baseline.py
  imbalance.py
  preprocessing.py
  sampling.py
  oversampling.py
  smote.py
