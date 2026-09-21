# 🚚 Delivery Intelligence & Sequence Analytics — Complete Technical Specification

> **An End-to-End, Production-Grade Machine Learning System for Last-Mile Express Logistics**  
> Features calibrated delivery duration estimation (ETA), prediction intervals ([P10, P90]), high-delay risk classification, empirical sequence advisories, FastAPI microservices, and interactive Streamlit analytics.

---

## 📑 Table of Contents
1. [Executive Overview & Domain Characterization](#-1-executive-overview--domain-characterization)
2. [End-to-End System Workflow Architecture](#-2-end-to-end-system-workflow-architecture)
3. [Data Leakage Audit & Scientific Remediation](#-3-data-leakage-audit--scientific-remediation)
4. [Feature Engineering & Point-in-Time Pipeline](#-4-feature-engineering--point-in-time-pipeline)
5. [Machine Learning Modeling & Uncertainty Calibration](#-5-machine-learning-modeling--uncertainty-calibration)
6. [Statistical Validation & Bootstrap Confidence Intervals](#-6-statistical-validation--bootstrap-confidence-intervals)
7. [Explainability (TreeSHAP) & Subgroup Error Analysis](#-7-explainability-treeshap--subgroup-error-analysis)
8. [Production Deployment (FastAPI & Streamlit)](#-8-production-deployment-fastapi--streamlit)
9. [Step-by-Step Execution Guide](#-9-step-by-step-execution-guide)

---

## 📌 1. Executive Overview & Domain Characterization

### 1.1 The Operational Problem
In last-mile express parcel logistics, dispatch efficiency and SLA fulfillment are determined by complex, compounding variables:
* **Wave Dispatch Congestion**: Couriers receive large batches (10–30 parcels) in wave dispatches and execute distribution runs over several hours.
* **Geospatial & Shift Friction**: Long routes and spatial repositioning between consecutive pickups introduce non-linear transit delays.
* **Sequence Lag Dynamics**: Delays in earlier deliveries during a shift can correlate with prolonged completion times for subsequent parcels.

### 1.2 Empirical Dataset Facts (Jilin Benchmark)
* **Dataset Scale**: 30,450 verified clean delivery orders across 57 couriers and 163 calendar days (May 12 to October 28).
* **Delivery Horizon**:
  - **Median Duration ($P_{50}$)**: **$175.0\text{ minutes}$ ($2.92\text{ hours}$)**
  - **75th Percentile ($P_{75}$)**: **$276.0\text{ minutes}$ ($4.60\text{ hours}$)**
  - **90th Percentile ($P_{90}$ High-Delay Threshold)**: **$379.6\text{ minutes}$ ($6.33\text{ hours}$)**
* **Operational Regime**: This dataset represents **e-commerce express parcel distribution** (e.g. Cainiao / SF Express / JD style operations), rather than 30-minute instant food delivery.

```
Distribution of Delivery Duration:
  < 1 hour:    12.06%  (3,789 orders)
  1 – 3 hours: 39.67% (12,463 orders)
  3 – 6 hours: 35.75% (11,229 orders)
  > 6 hours:   12.52%  (3,934 orders)
```

---

## 🏛️ 2. End-to-End System Workflow Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. INGESTION & DATA ENGINEERING                                                       │
 │    Raw Cainiao/Alibaba Express Logistics Data ──► GPS Sanitization & Filtering          │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 2. LEAKAGE-FREE POINT-IN-TIME FEATURE PIPELINE (scripts/03_ml_preprocessing.py)       │
 │    • Point-in-time completed task lookup: max(delivery_time) <= current accept_time    │
 │    • Active in-flight concurrent load: sum(delivery_time > current accept_time)        │
 │    • Shift context: daily_task_index, is_first_task_of_day, daily dispatch gap         │
 │    • Temporal Split: 70% Train (May-Aug) | 15% Val (Aug-Sep) | 15% Test (Sep-Oct)     │
 │    • Target High-Delay Threshold (P90 = 379.6m) fitted STRICTLY on Training partition   │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 3. AUTOMATED LEAKAGE & INTEGRITY TEST SUITE (tests/test_leakage.py)                   │
 │    • Pytest verification: zero future timestamps, monotonic splits, valid daily resets │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
 ┌───────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
 │ 4A. DURATION REGRESSION BENCHMARK         │ │ 4B. HIGH-DELAY CLASSIFICATION BENCHMARK  │
 │     (scripts/04_duration_regression_models)│ │     (scripts/05_high_delay_classification│
 │ • Baselines: Global / Courier / Bucket    │ │ • Baselines: Majority / Stratified Prior │
 │ • Champion: LightGBM Regressor            │ │ • Champion: LightGBM Classifier          │
 │   MAE: 93.34 min | R²: 0.2530             │ │   ROC-AUC: 0.7886 | PR-AUC: 0.3461       │
 │ • Quantile Regressors (P10, P50, P90)     │ │ • Probability Calibration (Isotonic/Platt│
 │   Empirical Coverage: 79.29%              │ │   ECE reduced 10x: 0.2899 ──► 0.0292     │
 └─────────────────────┬─────────────────────┘ └────────────────────┬─────────────────────┘
                       │                                            │
                       └──────────────────────┬─────────────────────┘
                                              │
                                              ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 5. STATISTICAL BOOTSTRAP & ADVISORY ENGINE (scripts/06_sequence_recommendation_engine) │
 │    • 1,000 Bootstrap iterations (95% CIs for MAE, R², ROC-AUC, PR-AUC, and Odds Ratios)│
 │    • Data-driven residual buffer: +175m (P80 overrun) / +259m (P90 overrun)            │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
 ┌───────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
 │ 6A. EXPLAINABILITY & SUBGROUP DIAGNOSTICS │ │ 6B. PRODUCTION SERVING & USER INTERFACE  │
 │     • TreeSHAP beeswarm & dependence      │ │ • FastAPI Microservice (:8000)           │
 │     • Subgroup error slicing by distance, │ │   (Single ETA, [P10, P90], Batch API)    │
 │       time of day, in-flight load, region │ │ • Streamlit Interactive Dashboard (:8501)│
 │     • Feature Ablation Benchmarks         │ │   (Real-Time Simulator, Batch Analyzer)  │
 └───────────────────────────────────────────┘ └──────────────────────────────────────────┘
```

---

## 🔬 3. Data Leakage Audit & Scientific Remediation

A comprehensive engineering audit discovered critical flaws in naive sequential implementations:

### 3.1 The `shift(1)` Target Leakage Mechanism
In wave dispatch logistics, couriers receive multiple packages at the exact same minute. 
* **Audit Finding**: In **$91.24\%$ of sequential pairs** ($28,610 / 31,358$), the preceding assigned package was still **unfinished** when the current package was assigned.
* **The Flaw**: Applying `groupby("courier_id")["delivery_duration_minutes"].shift(1)` assigned the *future completed duration* of a concurrent morning parcel to the current parcel, artificially inflating the correlation to $r = 0.377$ and causing an artifactual "$8\times$ delay risk" claim.
* **The Remediation**: Replaced with `duration_of_most_recently_COMPLETED_task_as_of_dispatch_time` ($\max(\text{delivery\_time}) \le \text{accept\_time}$).

### 3.2 Before vs. After Scientific Comparison

| Dimension | Legacy Leaky Setup (Archived in `/results/legacy_random_split/`) | Upgraded Production Setup (Active in `/results/`) | Scientific Rationale |
|---|---|---|---|
| **Split Method** | Random 80/20 Split | **Chronological 70% Train / 15% Val / 15% Test** | Prevents future shift data from training past predictions. |
| **Prior Duration Feature** | Naive `shift(1)` on target duration | **Strict point-in-time completed task lookup** | Restricts information access strictly to completed events. |
| **Shift Counters** | Continuous cumcount over 163 days | **`daily_task_index` reset per shift** | Restricts sequence counters to active daily shifts. |
| **Delay Threshold** | Computed on entire dataset ($384\text{m}$) | **Fitted strictly on 70% Train split ($379.6\text{m}$)** | Zero target distribution leakage into validation/test sets. |
| **Sequence Correlation** | $r = 0.377$ (Artifact) | **$r = 0.1334$ (95% CI: $[0.1036, 0.1607]$)** | True, non-leaked observational correlation. |
| **Delay Odds Ratio** | $8.0\times$ (Artifact) | **$1.66\times$ (95% CI: $[1.36, 2.01]$)** | Statistically validated $66\%$ increase in delay odds ($p < 0.001$). |

---

## ⚙️ 4. Feature Engineering & Point-in-Time Pipeline

The feature pipeline ([`scripts/03_ml_preprocessing.py`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/scripts/03_ml_preprocessing.py)) constructs 16 strictly pre-acceptance features:

1. **Spatial Features**:
   - `delivery_distance_km`: Geodesic Haversine distance between pickup and dropoff.
   - `previous_accept_distance_km`: Haversine transition distance from the prior pickup location.
2. **Workload & Concurrency Features**:
   - `active_inflight_tasks`: Count of accepted packages currently in flight for this courier.
   - `tasks_previous_1h` & `tasks_previous_3h`: Rolling orders accepted in prior 1h and 3h windows.
   - `daily_task_index`: Courier's 0-indexed order count for today's shift.
   - `is_first_task_of_day`: Binary indicator (1 if `daily_task_index == 0`).
   - `minutes_since_last_dispatch_today`: Time elapsed since the previous order accepted today.
3. **Point-in-Time Sequence Features**:
   - `duration_of_most_recently_completed_task`: Completed delivery duration of the courier's most recent delivered package.
   - `has_prior_completed_task`: Binary flag (1 if courier has finished $\ge 1$ delivery earlier today/historically).
   - `mins_since_recent_completed`: Time elapsed since that delivery was completed.
4. **Temporal Features**:
   - `accept_hour`, `accept_minute`, `accept_weekday`, `is_weekend`, `time_period` ("Morning", "Afternoon", "Evening", "Night").

---

## 🤖 5. Machine Learning Modeling & Uncertainty Calibration

### 5.1 Model A: Delivery Duration Regression (ETA)
*Evaluated on the Chronological Test Partition ($N = 4,568$).*

```
Regression Benchmark:
├── Baseline 1 (Courier Historical Median): MAE = 108.71m | R² = -0.2343
├── Baseline 2 (Distance x Hour Bucket):   MAE = 106.94m | R² = -0.1351
├── Baseline 3 (Global Median):            MAE = 106.24m | R² = -0.0728
├── Baseline 4 (Ridge Regression):         MAE = 103.67m | R² =  0.0730
├── XGBoost Regressor:                     MAE =  95.32m | R² =  0.2271
├── LightGBM Regressor (Champion):         MAE =  93.34m | R² =  0.2530  (RMSE: 122.00m)
└── Random Forest Regressor:               MAE =  92.36m | R² =  0.2575
```

### 5.2 Calibrated Uncertainty Prediction Intervals ([P10, P90])
Trained dedicated LightGBM Quantile Regressors ($\alpha = 0.10, 0.50, 0.90$):
* **Target Coverage**: $80.0\%$
* **Empirical Test Coverage**: **$79.29\%$** (Exemplary empirical calibration)
* **Median Interval Width**: **$277.16\text{ minutes}$ ($4.62\text{ hours}$)**

### 5.3 Model B: High-Delay Risk Classification ($>379.6\text{ min}$)
*Evaluated on Chronological Test Partition ($N = 4,568$, 10.99% delay rate).*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Majority Baseline (All 0)** | 0.8901 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1099 | 0.0979 |
| **Stratified Prior Baseline** | 0.8170 | 0.1152 | 0.0996 | 0.1068 | 0.5000 | 0.1099 | 0.0979 |
| **Logistic Regression (Balanced)** | 0.5464 | 0.1174 | 0.4801 | 0.1887 | 0.5194 | 0.1739 | 0.2499 |
| **Random Forest (Balanced)** | 0.7095 | 0.2184 | 0.6375 | 0.3254 | 0.7311 | 0.1953 | 0.1864 |
| **XGBoost (Weighted)** | 0.6929 | 0.2221 | 0.7171 | 0.3391 | 0.7809 | 0.3290 | 0.1943 |
| **LightGBM Classifier (Champion)** | **0.7040** | **0.2324** | **0.7351** | **0.3531** | **0.7886** | **0.3461** | **0.1906** |

### 5.4 Probability Calibration (Reliability Diagram)
* **Uncalibrated LightGBM**: $\text{ECE} = 0.2899$, $\text{Brier} = 0.1906$
* **Platt (Sigmoid) Calibrated**: $\text{ECE} = 0.0363$, $\text{Brier} = 0.0862$
* **Isotonic Calibrated (Champion)**: **$\text{ECE} = 0.0292$**, **$\text{Brier} = 0.0860$** (**$10\times$ error reduction**)

---

## 📊 6. Statistical Validation & Bootstrap Confidence Intervals

1,000 bootstrap resamples on the test partition established strict 95% Confidence Intervals:

```
Bootstrap 95% Confidence Intervals (Test Partition, 1,000 iterations):
├── Regression MAE:           93.37 min  (95% CI: [91.18, 95.58])
├── Regression RMSE:         121.94 min  (95% CI: [118.82, 124.97])
├── Regression R²:             0.2521    (95% CI: [0.2263, 0.2797])
├── Classification ROC-AUC:    0.7847    (95% CI: [0.7650, 0.8048])
├── Classification PR-AUC:     0.3152    (95% CI: [0.2793, 0.3514])
├── Non-leaked Sequence r:     0.1334    (95% CI: [0.1036, 0.1607])
├── Non-leaked Spearman rho:   0.1244    (95% CI: [0.0951, 0.1524])
└── High-Delay Odds Ratio:     1.66x     (95% CI: [1.3611, 2.0061])
```

### Multi-Split Generalization Benchmark

| Evaluation Split | Generalization Target | Regressor MAE (min) | Regressor $R^2$ | Classifier ROC-AUC | Classifier PR-AUC |
|---|---|---:|---:|---:|---:|
| **Temporal 70/15/15 (Production)** | **Chronological Future Generalization** | **93.34** | **0.2530** | **0.7886** | **0.3461** |
| **Courier-Grouped Holdout** | **Cold-Start (Unseen Couriers)** | 87.41 | -0.1117 | 0.7251 | 0.2185 |
| **Random 80/20** | **Uniform Interpolation Baseline** | 84.99 | 0.3128 | 0.8118 | 0.4012 |

---

## 🧠 7. Explainability (TreeSHAP) & Subgroup Error Analysis

### 7.1 TreeSHAP Global Feature Attribution
Top drivers of delivery duration (mean absolute SHAP impact):
1. **`delivery_distance_km`**: $\pm 28.43\text{ minutes}$
2. **`mins_since_recent_completed`**: $\pm 13.44\text{ minutes}$
3. **`accept_hour`**: $\pm 10.43\text{ minutes}$
4. **`duration_of_most_recently_completed_task`**: $\pm 9.32\text{ minutes}$
5. **`tasks_previous_3h`**: $\pm 7.06\text{ minutes}$
6. **`daily_task_index`**: $\pm 6.76\text{ minutes}$

### 7.2 Feature Ablation Study

| Configuration | Features | Regressor MAE (min) | Regressor $R^2$ | Classifier ROC-AUC | Classifier PR-AUC |
|---|---:|---:|---:|---:|---:|
| **1. Full Feature Matrix** | **16** | **93.61** | **0.2509** | **0.7774** | **0.3420** |
| 2. Ablated: No Sequence Lookups | 12 | 92.79 | 0.2654 | 0.7771 | 0.4089 |
| 3. Ablated: No Workload/Concurrency | 10 | 94.41 | 0.2369 | 0.7771 | 0.3251 |
| 4. Ablated: Distance + Temporal Only | 6 | 95.64 | 0.2233 | 0.7538 | 0.3933 |
| 5. Baseline: Distance Only | 1 | 100.18 | 0.0859 | 0.7501 | 0.3309 |

---

## 🚀 8. Production Deployment (FastAPI & Streamlit)

### 8.1 FastAPI REST Microservice (`api/main.py`)
* **`POST /predict/dispatch-evaluation`**: Full inference returning point ETA, $[P_{10}, P_{90}]$ intervals, calibrated delay probability, and sequence advisory.
* **`POST /predict/raw-gps`**: On-the-fly Haversine distance and temporal calculation from raw coordinate inputs.
* **`POST /predict/batch`**: High-throughput batch evaluation.
* **`GET /health`**: Operational readiness check verifying loaded model weights.
* **`GET /models/rules`**: Inspect empirical sequence advisory thresholds.

#### Benchmarked Latency (300 consecutive requests):
* **Single Order Latency**: Mean **$24.96\text{ ms}$**, Median **$24.99\text{ ms}$**, $P_{95}$ **$28.66\text{ ms}$**, $P_{99}$ **$31.30\text{ ms}$**
* **Throughput**: **$40.1\text{ QPS}$** per single worker process
* **Batch Efficiency**: **$20.97\text{ ms}$ per order** for 50-order batches

### 8.2 Streamlit Operations Dashboard (`app/delivery_intelligence_app.py`)
* **Tab 1: 🚀 Real-Time Dispatch Simulator**: Interactive dispatch parameters, calibrated ETA intervals, gauge charts, and sequence alerts.
* **Tab 2: 📁 Batch Dataset Risk Analyzer**: Ingestion of CSV/Parquet delivery datasets with schema standardization and batch prediction download.
* **Tab 3: 📊 Operational & Model Intelligence**: Dynamic visualizations bound to `/results/` JSON/CSVs (Bootstrap distributions, SHAP beeswarms, reliability diagrams, subgroup diagnostics).

---

## 💻 9. Step-by-Step Execution Guide

### 1. Run Leakage Tests
```powershell
pytest tests/test_leakage.py
```

### 2. Run the Full Pipeline
```powershell
# 1. Feature Preprocessing & Temporal Split
python scripts/03_ml_preprocessing.py

# 2. Train Regression Models & Quantile Intervals
python scripts/04_duration_regression_models.py

# 3. Train Classification Models & Probability Calibration
python scripts/05_high_delay_classification_models.py

# 4. Bootstrap Confidence Intervals & Advisory Engine
python scripts/06_sequence_recommendation_engine.py

# 5. TreeSHAP & Subgroup Analysis
python scripts/07_shap_explainability.py
python scripts/08_subgroup_error_analysis.py
python scripts/09_ablation_study.py
```

### 3. Launch FastAPI Backend
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Interactive Swagger docs available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.*

### 4. Launch Streamlit Operations App
*(In a separate terminal window)*
```powershell
streamlit run app/delivery_intelligence_app.py
```
*Web dashboard available at **[http://localhost:8501](http://localhost:8501)**.*
