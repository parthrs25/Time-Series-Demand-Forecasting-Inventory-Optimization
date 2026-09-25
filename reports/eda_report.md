# 📊 Comprehensive EDA & System Architecture Report
**Project:** Retail Multi-Store Hierarchical Demand Forecaster (Kaggle Walmart M5 Dataset)  
**Date:** August 2026  
**Primary Code Artifacts:** [`src/eda.py`](file:///c:/Users/user7/Desktop/Time_series_project/src/eda.py), [`src/hts.py`](file:///c:/Users/user7/Desktop/Time_series_project/src/hts.py), [`src/inference.py`](file:///c:/Users/user7/Desktop/Time_series_project/src/inference.py)  
**Secondary Runner Notebooks:** [`notebooks/01_data_ingestion_and_eda.ipynb`](file:///c:/Users/user7/Desktop/Time_series_project/notebooks/01_data_ingestion_and_eda.ipynb), [`notebooks/02_data_preprocessing_and_feature_engineering.ipynb`](file:///c:/Users/user7/Desktop/Time_series_project/notebooks/02_data_preprocessing_and_feature_engineering.ipynb), [`notebooks/03_model_training_and_evaluation.ipynb`](file:///c:/Users/user7/Desktop/Time_series_project/notebooks/03_model_training_and_evaluation.ipynb)

---

## 1. Executive Summary & Dataset Metadata

This document provides a detailed statistical audit, exploratory data analysis (EDA), and system architecture breakdown of the **Kaggle Walmart M5 Forecasting Dataset**. The objective is to identify key variance-driving signals—such as zero inflation, seasonality, promotional elasticity, calendar shock events, and cross-store heterogeneity—to guide our MLOps feature engineering, hierarchical reconciliation, and supply chain inventory decision pipeline.

### 📈 Global Dataset Summary Statistics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Dataset Source** | Kaggle Walmart M5 | Real production retail sales data from Walmart US stores |
| **Total Time Series** | **30,490** | Unique `item_id` × `store_id` combinations |
| **Observation Horizon** | **1,913 days** | Daily records from `d_1` (Jan 29, 2011) to `d_1913` (Apr 24, 2016) |
| **Total Historical Records** | **~58.3 Million** | 30,490 series × 1,913 days panel matrix |
| **Total Units Sold** | **65,695,409** | Cumulative unit volume across all stores and items |
| **Global Zero Sales Ratio** | **68.20%** | Proportion of daily item-store combinations with 0 sales |
| **States Represented** | **3** | California (`CA`), Texas (`TX`), Wisconsin (`WI`) |
| **Stores Represented** | **10** | `CA_1`, `CA_2`, `CA_3`, `CA_4`, `TX_1`, `TX_2`, `TX_3`, `WI_1`, `WI_2`, `WI_3` |
| **Product Categories** | **3** | `FOODS`, `HOBBIES`, `HOUSEHOLD` |
| **Product Departments** | **7** | `FOODS_1`, `FOODS_2`, `FOODS_3`, `HOBBIES_1`, `HOBBIES_2`, `HOUSEHOLD_1`, `HOUSEHOLD_2` |
| **Unique Item SKUs** | **3,049** | Individual retail product items |

---

## 2. Hierarchical Aggregation Levels

The dataset is organized hierarchically across 10 distinct aggregation levels. Our forecasting pipeline must produce consistent predictions across all levels:

```
[ Level 0: Total Corporate Sales (All Stores & Items) ]
            └── [ Level 1: State Aggregation (CA, TX, WI) ]
                 └── [ Level 2: Store Aggregation (10 Stores) ]
                      └── [ Level 3: Category Aggregation (FOODS, HOBBIES, HOUSEHOLD) ]
                           └── [ Level 4: Department Aggregation (7 Departments) ]
                                └── [ Level 5: Item SKU Level (30,490 Time Series) ]
```

---

## 3. Detailed 11-Part EDA Process & Diagnostic Insights

---

### Module 1: Univariate Sales Distribution & Price Spread
* **Diagnostic Figure:** [`reports/figures/eda_1_univariate_sales_distribution.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_1_univariate_sales_distribution.png) & [`reports/figures/eda_1_univariate_price_distribution.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_1_univariate_price_distribution.png)

#### Methodology & Findings
* Analyzed the frequency distribution of daily sales counts and product sell prices ($ USD) across categories.
* **Zero Sales Ratio:** **68.20%** of all observations are zero. Non-zero daily sales are heavily right-skewed, following a Tweedie / Poisson-like count distribution (median non-zero sale = 2 units, max single-day sale = 360 units).
* **Price Spread:** `FOODS` items have low median prices ($1.50–$3.50) with narrow variance, whereas `HOBBIES` and `HOUSEHOLD` items range from $0.50 up to $25.00+.

#### Modeling Significance
> [!IMPORTANT]
> Standard Mean Squared Error (MSE) loss assumes Gaussian errors and will perform poorly on zero-inflated counts by predicting negative or over-smoothed fractional numbers. We must use **Tweedie Loss** ($\text{power} \in [1.1, 1.5]$) or zero-inflated GBDT objectives.

---

### Module 2: Bivariate Store, State & Category Performance
* **Diagnostic Figure:** [`reports/figures/eda_2_bivariate_store_state_sales.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_2_bivariate_store_state_sales.png) & [`reports/figures/eda_2_bivariate_snap_impact.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_2_bivariate_snap_impact.png)

#### Methodology & Findings
* **State Volume:** California (`CA`) is the largest market (~45% of total volume), followed by Texas (`TX`, ~30%) and Wisconsin (`WI`, ~25%).
* **Store Performance:** `CA_3` is the top-performing supercenter, outperforming `CA_4` by more than 2.2x.
* **Policy Impact of SNAP Days:** On active SNAP (food stamp) benefit days (days 1–10 of each month in CA), `FOODS` daily sales surge by **+16.2%** compared to non-SNAP days. Non-food categories show minimal change.

#### Modeling Significance
> [!TIP]
> Exogenous binary indicators (`snap_CA`, `snap_TX`, `snap_WI`) are highly informative features for predicting food demand spikes during the first 10 days of every month.

---

### Module 3: Multivariate Time Series Decomposition & Autocorrelation
* **Diagnostic Figure:** [`reports/figures/eda_3_multivariate_macro_trend.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_3_multivariate_macro_trend.png) & [`reports/figures/eda_3_multivariate_acf_pacf.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_3_multivariate_acf_pacf.png)

#### Methodology & Findings
* **5-Year Macro Trajectory:** Steady upward growth from 2011 to 2016 with clear annual seasonality.
* **Christmas Shock Signal:** Sales drop to near zero every year on **December 25** (Walmart stores are closed).
* **Autocorrelation (ACF/PACF):** Extremely strong autocorrelation spikes at **lags 7, 14, 21, and 28 days** ($r > 0.76$), confirming strong weekly cyclic recurrence.

#### Modeling Significance
> [!NOTE]
> To forecast 28 days ahead without data leakage, all lag features must be shifted by at least 28 days (`shift(28)`). The strong correlation at $t-28, t-35, t-42$ validates this feature design.

---

### Module 4: Composite Price Elasticity & Markdown Discount Tiers
* **Diagnostic Figure:** [`reports/figures/eda_4_composite_price_elasticity.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_4_composite_price_elasticity.png)

#### Methodology & Findings
* Constructed `discount_pct = (max_historical_price - sell_price) / max_historical_price * 100` and binned into 4 markdown tiers (`0-5%`, `5-15%`, `15-30%`, `30%+`).
* `FOODS` items are relatively inelastic (volume rises modestly under markdowns), whereas `HOBBIES` items exhibit **high elasticity (+68% volume lift under >30% clearance markdowns)**.

---

### Module 5: Composite Payday & Multi-State SNAP Concurrency
* **Diagnostic Figure:** [`reports/figures/eda_5_composite_payday_cashflow_cycles.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_5_composite_payday_cashflow_cycles.png)

#### Methodology & Findings
* Constructed `is_payday` (1st and 15th bimonthly paycheck windows) and `snap_concurrency` ($0 \dots 3$ active states).
* Total daily sales spike by **+12.4% during payday windows** and by **+24.8% when all 3 states run concurrent SNAP distributions**.

---

### Module 6: Syntetos-Boylan Demand Categorization ($\text{ADI}$ vs. $CV^2$)
* **Diagnostic Figure:** [`reports/figures/eda_6_composite_demand_classification_grid.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_6_composite_demand_classification_grid.png)

#### Methodology & Findings
* Computed **Average Demand Interval ($\text{ADI}$)** and **Squared Coefficient of Variation ($CV^2$)** across series:
  * **Smooth ($\text{ADI} < 1.32, CV^2 < 0.49$):** 34% of series (mostly `FOODS`).
  * **Intermittent ($\text{ADI} \ge 1.32, CV^2 < 0.49$):** 28% of series.
  * **Erratic ($\text{ADI} < 1.32, CV^2 \ge 0.49$):** 19% of series.
  * **Lumpy ($\text{ADI} \ge 1.32, CV^2 \ge 0.49$):** 19% of series (mostly `HOBBIES` / `HOUSEHOLD`).

---

### Module 7: Department Basket Wallet-Share Mix
* **Diagnostic Figure:** [`reports/figures/eda_7_composite_dept_basket_mix.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_7_composite_dept_basket_mix.png)

---

### Module 8: Multi-Store & Category Heterogeneity (10-Store Facet Grid)
* **Diagnostic Figure:** [`reports/figures/eda_8_store_department_heterogeneity_grid.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_8_store_department_heterogeneity_grid.png)

---

### Module 9: Event Shock Spectrum & Holiday 14-Day Ramp Grid
* **Diagnostic Figure:** [`reports/figures/eda_9_event_type_shock_spectrum_grid.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_9_event_type_shock_spectrum_grid.png)

---

### Module 10: Multi-Lag Autocorrelation Matrix across Hierarchies
* **Diagnostic Figure:** [`reports/figures/eda_10_lag_correlation_variance_grid.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_10_lag_correlation_variance_grid.png)

---

### Module 11: Zero-Streak Run-Length & Item Lifecycle Adoption Grid
* **Diagnostic Figure:** [`reports/figures/eda_11_zero_streaks_item_lifecycle_grid.png`](file:///c:/Users/user7/Desktop/Time_series_project/reports/figures/eda_11_zero_streaks_item_lifecycle_grid.png)

---

## 4. Summary of Feature Engineering Directives Derived from EDA

Based on our empirical EDA findings, our downstream feature pipeline in `src/feature_engineering.py` implements:

1. **Shifted Lag Features ($t-28$ Safety Margin):** Lags at $t-28, t-29, t-35, t-42, t-56, t-364$.
2. **Shifted Rolling Window Statistics:** Rolling mean and std over 7, 14, 28, 56, and 180 days.
3. **Price & Discount Features:** `sell_price / max_price`, `discount_pct`, `sell_price / rolling_4wk_price`.
4. **Calendar & Event Proximity:** `days_until_next_event`, `days_since_last_event`, `is_payday`, `snap_concurrency`.
5. **Item Lifecycle Features:** `days_since_first_sale`, `rolling_zero_ratio_28d`.
6. **Loss Function:** LightGBM trained with **Tweedie Loss** ($\text{power} = 1.15$).

---

## 5. System Architecture: HTS Reconciliation & Supply Chain Engine

### 🌲 `src/hts.py` — Hierarchical Time Series Reconciliation
`src/hts.py` resolves the fundamental challenge of hierarchical demand forecasting: guaranteeing that SKU-level forecasts aggregate **100% consistently** to upper corporate tiers.

* **Hierarchical Levels Reconciled:**
  - **Level 0 (National):** Total corporate sales forecast (CFO budget).
  - **Level 1 (State):** `CA`, `TX`, `WI` aggregate forecasts (Regional logistics).
  - **Level 2 (Store):** 10 Store forecasts (Store manager planning).
  - **Level 3 (Category):** `FOODS`, `HOBBIES`, `HOUSEHOLD` (Category buyer negotiations).
  - **Level 4 (Department):** 7 Department forecasts.
  - **Level 5 (Item SKU):** 30,490 individual item-store time series.
* **Mechanism:** Constructs structural matrix $S$ and performs **Bottom-Up & MinT Reconciliation**, eliminating cross-level forecast discrepancies.

---

### 📦 `src/inference.py` — Production Supply Chain Inventory Reorder Engine
`src/inference.py` translates daily demand forecasts ($\mu_{\text{daily demand}}$) and demand uncertainty ($\sigma_{\text{demand}}$) into **automated operational replenishment decisions**.

* **1. Safety Stock (SS):** Buffer stock to absorb demand spikes during supplier lead times:
  $$\text{Safety Stock} = Z \times \sigma_{\text{demand}} \times \sqrt{\text{Lead Time}}$$
  *(where $Z = 1.645$ for a 95% target in-stock service level)*.

* **2. Reorder Point (ROP):** The exact inventory threshold that triggers a purchase order:
  $$\text{Reorder Point (ROP)} = (\mu_{\text{daily demand}} \times \text{Lead Time}) + \text{Safety Stock}$$

* **3. Automated Replenishment Alerts:**
  - 🔴 **`CRITICAL_STOCKOUT_RISK`:** On-hand inventory < Safety Stock $\rightarrow$ Emergency expedited shipment.
  - 🟡 **`PLACE_PURCHASE_ORDER`:** On-hand inventory $\le$ ROP $\rightarrow$ Automatically reorder $N$ units.
  - 🟢 **`SUFFICIENT_STOCK`:** Inventory level healthy $\rightarrow$ No order required.
