# Real-Time Credit Card Fraud Detection & Prevention System

> A production-grade, leak-free machine learning pipeline designed to detect fraudulent financial transactions under severe class imbalance while satisfying strict sub-100ms operational latency SLAs and regulatory Explainable AI (XAI) mandates.

---

## 📋 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Literature Review & Research Gaps](#-literature-review--research-gaps)
3. [System Architecture & Tech Stack](#-system-architecture--tech-stack)
4. [Methodology & Implementation](#-methodology--implementation)
5. [Key Results & Performance Metrics](#-key-results--performance-metrics)
6. [Repository Structure](#-repository-structure)
7. [Dataset Source](#-dataset-source)
8. [Getting Started & Installation](#-getting-started--installation)
9. [Execution Guide](#-execution-guide)

---

## 🚨 Problem Statement

In modern digital payment ecosystems, financial institutions face two critical operational challenges:
1. **Extreme Class Imbalance:** Fraudulent transactions account for less than $0.3\%$ of total transaction volumes. Standard classifiers fail by predicting the majority class (legitimate), yielding misleadingly high accuracy while missing almost all fraud instances.
2. **Strict Latency Constraints (SLA):** Payment authorization gateways require real-time risk scoring decisions within **$<100\text{ ms}$** to prevent transaction bottlenecks or checkout failures.
3. **Regulatory Compliance (XAI):** Financial regulators enforce strict transparency rules ("Right to Explanation"), prohibiting black-box models that cannot justify why a transaction was blocked.

---

## 📚 Literature Review & Research Gaps

### Literature Evolution
Prior research progressed from static, rule-based threshold systems to classical machine learning algorithms (Logistic Regression, SVM, Decision Trees). Recently, Gradient-Boosted Decision Trees (GBDTs) like XGBoost and LightGBM have emerged as state-of-the-art standards for tabular financial data.

### Identified Research Gaps Addressed in This Study
* **Methodological Data Leakage:** Existing studies often apply over-sampling techniques (like SMOTE) globally across the entire dataset before train-test splitting, leading to test-set leakage and over-optimistic performance metrics.
* **Neglect of Latency SLAs:** Most academic literature focuses purely on offline accuracy metrics (ROC-AUC) without benchmarking single-sample inference latency required for live production APIs.
* **Misleading Evaluation Metrics:** Using standard Accuracy or ROC-AUC on highly imbalanced data masks poor minority-class recognition. Precision-Recall AUC (PR-AUC) and Fraud Recall must be prioritized.
* **Lack of Integrated Explainability:** High-performing models are rarely equipped with native auditing tools to satisfy banking regulatory frameworks.

---

## ⚙️ System Architecture & Tech Stack

The project is structured as a modular, production-ready Python application rather than a collection of isolated exploratory notebooks.

* **Language:** Python 3.9+
* **Core Libraries:** `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`, `shap`, `pandas`, `numpy`, `joblib`, `matplotlib`
* **Datasets:** PaySim (Synthetic Mobile Money Simulation) & ULB European Credit Card Logs.

---

## 🛠️ Methodology & Implementation

Our end-to-end pipeline (`src/` modules) executes the following sequential stages:

1. **Data Ingestion & High-Risk Filtering (`src/data_loader.py`):** Isolates high-risk transaction types (`TRANSFER` and `CASH_OUT`) where fraudulent behavior exclusively occurs.
2. **Domain-Specific Feature Engineering (`src/preprocessing.py`):** Derives crucial financial balance error deltas to capture account takeover anomalies:
   $$\text{errorBalanceOrig} = \text{newbalanceOrig} + \text{amount} - \text{oldbalanceOrg}$$
   $$\text{errorBalanceDest} = \text{oldbalanceDest} + \text{amount} - \text{newbalanceDest}$$
3. **Robust Scaling & Leak-Free Resampling:** Applies Interquartile Range (IQR) normalization (`RobustScaler`) and enforces a **Strictly Train-Only 10% Targeted SMOTE** strategy ($\alpha = 0.10$) to prevent data leakage.
4. **Cost-Sensitive Model Training (`src/models.py`):** Trains an optimized XGBoost classifier using histogram-based tree optimization (`tree_method='hist'`) and dynamic class weighting (`scale_pos_weight`).
5. **Evaluation & Explainability (`src/evaluation.py`):** Benchmarks single-transaction inference latency, computes PR-AUC/Recall metrics, and generates SHAP (`TreeExplainer`) global feature attributions.

---

## 📊 Key Results & Performance Metrics

Evaluated on large-scale transaction hold-out test splits, the system achieved the following production benchmarks:

| Evaluation Metric | Achieved Result | Production Target / SLA | Status |
| :--- | :---: | :---: | :---: |
| **Precision-Recall AUC (PR-AUC)** | **0.9792 - 0.9892** | Maximized | ✅ **Optimal** |
| **Fraud Recall (Sensitivity)** | **98.25% - 99.27%** | $> 95.0\%$ | ✅ **Passed** |
| **False Positive Rate (FPR)** | **$< 0.004\%$** | $< 0.01\%$ | ✅ **Passed** |
| **Single-Transaction Latency ($\tau$)** | **$\sim 1.24\text{ ms}$** | $< 100\text{ ms}$ | ✅ **Passed (~80x Faster)** |

---

## 📥 Dataset Source

* **PaySim Mobile Money Financial Dataset:** The primary raw dataset used for this research can be downloaded from Kaggle: [Kaggle - PaySim Synthetic Financial Datasets For Fraud Detection]([https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)).
* **ULB European Credit Card Fraud Dataset:** Secondary benchmark dataset used for validation.

---

## 🗂️ Repository Structure

```text
ResearchProject/
├── data/
│   └── raw/                  # Download raw PaySim_log.csv from Kaggle and place it here
├── notebooks/                # Exploratory Jupyter notebooks
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_imbalance_handling_smote.ipynb
│   ├── 03_model_training_xgboost_lgbm.ipynb
│   └── 04_latency_and_evaluation.ipynb
├── src/                      # Reusable production modules
│   ├── data_loader.py        # Data ingestion
│   ├── preprocessing.py      # Cleaning, engineering, scaling & SMOTE
│   ├── models.py             # Cost-sensitive model training
│   └── evaluation.py         # Metrics, latency benchmark & SHAP visuals
├── models/                   # Saved model binaries (.pkl)
├── reports/
│   └── figures/              # Exported SHAP summary plots & PR curves
├── main.py                   # Single-command execution entry point
└── requirements.txt          # Project dependencies
