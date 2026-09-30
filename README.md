# Telco Customer Churn — Imbalanced Learning & Neural Network Classification

A reproducible project for studying class imbalance, ANN classification, and churn decision thresholds.

## Project Status

**Day 8 of 10 — decision-threshold analysis**

The project now compares multiple probability cutoffs on one trained ANN using an untouched stratified test split.

## Workflow

1. Project structure and cleanup — complete
2. Leakage-safe preprocessing — complete
3. Baseline ANN + evaluation — complete
4. Class-imbalance analysis — complete
5. Random under-sampling — complete
6. Random over-sampling — complete
7. SMOTE comparison — complete
8. Threshold and metric analysis — complete
9. Reproducibility and code cleanup
10. Final documentation and completion checks

## Experiments

- `src/sampling.py`: random under-sampling
- `src/oversampling.py`: random over-sampling
- `src/smote.py`: leakage-safe SMOTE
- `src/thresholds.py`: compare churn-probability thresholds with accuracy, precision, recall, F1, ROC-AUC, and confusion-matrix counts

All experiments keep the test split untouched.

## Usage

Import `train_baseline`, `train_undersampled`, `train_oversampled`, `train_smote`, and `evaluate_thresholds` from `src`.

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
  thresholds.py
