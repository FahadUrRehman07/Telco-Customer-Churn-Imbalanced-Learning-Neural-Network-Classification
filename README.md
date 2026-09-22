# Telco Customer Churn — Imbalanced Learning & Neural Network Classification

A reproducible machine-learning project for predicting customer churn while studying the effect of class imbalance on model evaluation and decision-making.

## Project Status

**Day 1 of 10 — Project structure and notebook cleanup**

This repository is being developed as a 10-step portfolio project. Each step will add one meaningful improvement to the modeling workflow, evaluation quality, reproducibility, or documentation.

## Planned Workflow

1. Project structure and notebook cleanup
2. Leakage-safe train/test preprocessing pipeline
3. Baseline neural-network model and evaluation
4. Class-imbalance analysis
5. Random under-sampling comparison
6. Over-sampling comparison
7. SMOTE comparison
8. Threshold and metric analysis focused on precision, recall, and F1-score
9. Reproducibility, requirements, and code cleanup
10. Final documentation and project completion checks

## Intended Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── telco_customer_churn_imbalanced_learning.ipynb
├── src/
│   ├── data_preparation.py
│   ├── models.py
│   └── evaluation.py
└── reports/
    └── figures/
```

## Core Questions

- How much does class imbalance affect churn detection?
- Which resampling strategy best improves minority-class recall?
- How should precision, recall, F1-score, and threshold choice guide model selection?
- Can the final workflow remain leakage-safe and reproducible?

## Dataset

The project uses a Telco Customer Churn dataset. Raw data should be kept outside version control unless its license and redistribution terms permit committing it.

## Author

Fahad Ur Rehman
