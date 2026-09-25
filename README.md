# 🛒 Hierarchical Retail Demand Forecasting & Inventory Optimization MLOps Engine

[![CI Pipeline](https://github.com/AP7803/Hierarchical-Time-Series-Demand-Forecasting-Inventory-Optimization/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/AP7803/Hierarchical-Time-Series-Demand-Forecasting-Inventory-Optimization/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)

An enterprise-grade, end-to-end **Hierarchical Time Series (HTS)** demand forecasting and supply chain inventory replenishment engine built on the Kaggle Walmart M5 dataset (**30,490 daily time series across 10 supercenters, 25.76M rows**).

---

## 📌 Architecture Overview

```
                          ┌───────────────────────────┐
                          │   Raw Walmart M5 Ingest   │
                          │   (30,490 Series, 1913d)  │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Memory Downcasting Engine │
                          │   (80% RAM Reduction)     │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ 48 Feature Engineering    │
                          │ (Lags t-28, EWMA, Elastic)│
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Multi-Model Training      │
                          │ (Tweedie Loss, Optuna)    │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ 6-Level HTS Reconciliation│
                          │ (Bottom-Up Matrix Coher.) │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Supply Chain Decision Hub │
                          │ (Safety Stock & ROP @95%) │
                          └─────────────┬─────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
        ┌─────────────────────────┐           ┌─────────────────────────┐
        │ Streamlit Merchant Hub  │           │   CI/CD & MLflow Hub    │
        │ (Interactive Simulator) │           │ (Tracking & Automated)  │
        └─────────────────────────┘           └─────────────────────────┘
```

---

## 🎯 Key Features

1. **Intermittent Zero-Inflated Modeling:**
   - Handles **68.2% zero-sales days** using Compound Poisson-Gamma **Tweedie Loss ($p=1.35$)** to eliminate negative predictions.
2. **Leakage-Free Feature Engineering (48 Features):**
   - Autoregressive lags ($t-28$ to $t-364$), Exponential Moving Averages (**EWMA 7/28**), rolling volatility, price discount cross-elasticities, and state SNAP policy concurrence.
3. **6-Tier Hierarchical Reconciliation (HTS):**
   - Mathematically reconciles forecasts across National $\rightarrow$ State $\rightarrow$ Store $\rightarrow$ Category $\rightarrow$ Department $\rightarrow$ Item SKU levels.
   - **National Macro Accuracy:** **95.8% (4.23% WAPE)**
   - **SKU-Level Accuracy Gain:** **+15.7%** over naive seasonal baselines.
4. **Automated Supply Chain Inventory Replenishment:**
   - Computes statistical **Safety Stock ($SS$)** and **Reorder Points ($ROP$)** at a **95% target in-stock service level ($Z = 1.645$)**:
     $$\text{Safety Stock} = Z \times \sigma_{\text{lead}} \times \sqrt{L}$$
     $$\text{Reorder Point (ROP)} = (\mu_{\text{daily}} \times L) + \text{Safety Stock}$$
5. **Production MLOps Stack:**
   - **MLflow** experiment tracking (parameters, metrics, model artifacts).
   - **Streamlit** merchant command center with what-if promotional simulation and one-click purchase order CSV export.
   - **Docker** containerization & **GitHub Actions CI/CD** with automated Pytest verification.

---

## 📊 Benchmark Results

| Hierarchy Level | Series Count | Total Units (28d) | Naive Baseline WAPE (%) | LightGBM Tweedie WAPE (%) | Accuracy ($100 - \text{WAPE}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Level 0: National Total** | 1 | 1,183,626 | 5.76% | **4.23%** | **95.77%** 🥇 |
| **Level 1: State Level (CA, TX, WI)** | 3 | 1,183,626 | 6.99% | **5.31%** | **94.69%** |
| **Level 2: Store Level (10 Supercenters)** | 10 | 1,183,626 | 8.48% | **6.47%** | **93.53%** |
| **Level 3: Category $\times$ Store** | 30 | 1,183,626 | 10.59% | **8.17%** | **91.83%** |
| **Level 4: Department $\times$ Store** | 70 | 1,183,626 | 12.59% | **9.77%** | **90.23%** |
| **Level 5: Individual Item SKU** | 30,490 | 1,183,626 | 90.74% | **75.05%** | **+15.69% Boost** 🚀 |

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/AP7803/Hierarchical-Time-Series-Demand-Forecasting-Inventory-Optimization.git
cd Hierarchical-Time-Series-Demand-Forecasting-Inventory-Optimization

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
```bash
# Execute the master pipeline orchestrator (Ingestion -> Preprocessing -> Features -> Train -> HTS -> Inference)
python run_pipeline.py --stage all
```

### 3. Launch the Streamlit Merchant Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Unit & Integration Tests
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t retail-demand-forecaster .

# Run container
docker run -p 8501:8501 retail-demand-forecaster
```

---

## 📂 Repository Structure

```
├── .github/workflows/         # CI/CD Workflows (GitHub Actions)
│   ├── ci_pipeline.yml
│   └── cd_pipeline.yml
├── config/                    # Central Configuration Management
│   └── config.yaml
├── src/                       # Production Modular Pipeline
│   ├── data_ingestion.py      # Automated dataset downloader
│   ├── eda.py                 # 11-part diagnostic exploratory suite
│   ├── preprocessing.py       # Memory downcasting & long panel melt
│   ├── feature_engineering.py # Lags, EWMA, rolling stats, price elasticities
│   ├── train.py               # LightGBM Tweedie training & MLflow tracking
│   ├── hts.py                 # 6-Level Hierarchical Reconciliation
│   ├── evaluate.py            # Multi-level WAPE & RMSE evaluation audit
│   └── inference.py           # Production Safety Stock & ROP supply chain engine
├── notebooks/                 # Stage-wise Google Colab & GPU Notebooks
│   ├── 01_data_ingestion_and_eda.ipynb
│   ├── 02_data_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_training_and_evaluation.ipynb
│   └── colab_gpu_multi_model_tuning_pipeline.ipynb
├── tests/                     # Automated Pytest Suite
│   └── test_pipeline.py
├── app.py                     # Interactive Streamlit Merchant Dashboard
├── run_pipeline.py            # Master CLI Pipeline Orchestrator
├── Dockerfile                 # Containerization definition
├── requirements.txt           # Python dependency specifications
└── README.md
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.