# 🚚 Delivery Intelligence & Sequence Analytics — Technical Overview

> **Production-Grade Last-Mile Express Logistics AI & Point-in-Time Sequence Analytics**  
> Calibrated delivery duration estimation (ETA), prediction intervals ([P10, P90]), high-delay risk classification, and empirical sequence dispatch advisories.

---

## 📌 1. Project Overview & Operational Domain

This system provides machine learning intelligence for **last-mile e-commerce express parcel delivery** (e.g., Cainiao / SF Express / JD style operations). 

### Operational Domain Realities:
* **Wave Dispatch Execution**: Couriers receive batch dispatches of 10–30 parcels in morning and afternoon distribution waves.
* **Duration Distribution**: Empirical median duration is **$175.0\text{ minutes}$ ($2.92\text{ hours}$)**, with $P_{75} = 276.0\text{m}$ and $P_{90} = 379.6\text{m}$ ($6.33\text{h}$).
* **Dataset Scale**: 30,450 verified clean delivery dispatches across 57 couriers and 163 calendar days.

```
Distribution of Delivery Duration:
< 1 hour:    12.06%  (3,789 orders)
1 – 3 hours: 39.67% (12,463 orders)
3 – 6 hours: 35.75% (11,229 orders)
> 6 hours:   12.52%  (3,934 orders)
```

---

## 🔬 2. Empirical Leakage Remediation & Before/After Audit

A rigorous engineering audit identified that naive feature implementations (e.g. `shift(1)` on target duration across concurrent batch parcels) leaked future delivery completion times, creating artificial correlations. The system was completely remediated using **strict point-in-time lookups**.

| Dimension | Legacy Leaky Setup (Archived in `/results/legacy_random_split/`) | Production Point-in-Time Setup (Active in `/results/`) | Senior Engineering Takeaway |
|---|---|---|---|
| **Split Strategy** | Random 80/20 Uniform Split | **Chronological 70% Train / 15% Val / 15% Test** | Prevents future test data from training past predictions. |
| **Prior Duration Feature** | `shift(1)` on raw duration (Leaked future duration in 91.24% of pairs) | **`duration_of_most_recently_COMPLETED_task_as_of_dispatch_time`** | Strictly restricts feature access to events finished $\le \text{accept\_time}$. |
| **Daily Context** | Monotonic cumcount over 163 days | **`daily_task_index` & `is_first_task_of_day` reset per shift** | Restricts sequence counters to courier's active daily shift. |
| **High-Delay Threshold** | Computed globally before split ($384\text{m}$) | **Fit strictly on 70% Train partition ($379.6\text{m}$)** | Zero target distribution leakage into validation/test splits. |
| **Sequence Correlation** | $r = 0.377$ (Leakage artifact) | **$r = 0.1334$ (95% CI: $[0.1036, 0.1607]$)** | Real, statistically validated observational association. |
| **Delay Odds Ratio** | $8.0\times$ (Leakage artifact) | **$1.66\times$ (95% CI: $[1.36, 2.01]$)** | A preceding delay increases next delay odds by ~66% ($p < 0.001$). |

---

## 🤖 3. Machine Learning Benchmarks & Validation Results

All metrics below are dynamically sourced from [`results/`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/results/) under strict chronological testing.

### Model A: Delivery Duration Regression (ETA)
*Evaluated on Chronological Test Partition ($N = 4,568$).*

| Algorithm | Test MAE (min) | Test RMSE (min) | Test $R^2$ | Status |
|---|---:|---:|---:|:---:|
| Courier Historical Median Baseline | 108.71 | 156.82 | -0.2343 | Naive Baseline |
| Distance $\times$ Hour Bucket Median Baseline | 106.94 | 150.39 | -0.1351 | Spatial-Temporal Baseline |
| Global Median Baseline | 106.24 | 146.20 | -0.0728 | Aggregate Baseline |
| Ridge Regression Baseline | 103.67 | 135.90 | 0.0730 | Linear Baseline |
| XGBoost Regressor | 95.32 | 124.09 | 0.2271 | Tree Model |
| **LightGBM Regressor** | **93.34** | **122.00** | **0.2530** | 🏆 **Champion Regressor** |
| Random Forest Regressor | 92.36 | 121.63 | 0.2575 | Tree Model |

*1,000-sample bootstrap 95% CI for Champion LightGBM: MAE **$[91.18\text{m}, 95.58\text{m}]$**, $R^2$ **$[0.2263, 0.2797]$**.*

### Model B: High-Delay Risk Classification ($>379.6\text{ min}$)
*Evaluated on Chronological Test Partition ($N = 4,568$, 10.99% positive cases).*

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Status |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Majority Class Baseline (Always 0) | 0.8901 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1099 | Baseline |
| Stratified Prior Baseline | 0.8170 | 0.1152 | 0.0996 | 0.1068 | 0.5000 | 0.1099 | Baseline |
| Logistic Regression (Balanced) | 0.5464 | 0.1174 | 0.4801 | 0.1887 | 0.5194 | 0.1739 | Linear Baseline |
| Random Forest (Balanced) | 0.7095 | 0.2184 | 0.6375 | 0.3254 | 0.7311 | 0.1953 | Benchmarked |
| XGBoost (Weighted) | 0.6929 | 0.2221 | 0.7171 | 0.3391 | 0.7809 | 0.3290 | Benchmarked |
| **LightGBM Classifier** | **0.7040** | **0.2324** | **0.7351** | **0.3531** | **0.7886** | **0.3461** | 🏆 **Champion** |

*Validation-tuned optimal decision threshold ($T = 0.20$): Precision **0.3205**, Recall **0.5319**, F1-Score **0.4000**.*

### Model C: Gradient Boosted Trees vs. Deep Sequential Neural Network (PyTorch)
*Evaluates whether Deep Learning improves upon Tree-based methods on point-in-time sequential dispatch features.*

| Model Architecture | Model Paradigm | Test MAE (min) | Test $R^2$ | ROC-AUC | PR-AUC | Inference Speed (per order) |
|---|---|---:|---:|---:|---:|---:|
| **LightGBM (Point-in-Time GBDT)** | **Gradient Boosted Trees** | **93.34** | **0.2530** | **0.7886** | **0.3461** | **0.050 ms** |
| Deep Sequential DeliveryNet (PyTorch) | Deep Neural Network (Huber+BCE) | 103.00 | -0.0232 | 0.5892 | 0.1431 | 0.014 ms |

> [!NOTE]
> **Key Empirical Finding**: Tree-based ensembles (LightGBM/XGBoost) significantly outperform Deep Neural Networks on tabular sequential dispatch data ($+27.6\%\text{ higher } R^2$ and $+0.20\text{ higher ROC-AUC}$). Decision trees naturally partition non-linear spatial bounds and discrete shift resets without susceptibility to gradient vanishing over sparse categorical embeddings.

---


## 🎯 4. Uncertainty Quantification & Probability Calibration

1. **Calibrated Quantile Prediction Intervals ($[P_{10}, P_{90}]$)**:
   - LightGBM quantile regressors deliver an **empirical test coverage of $79.29\%$** (matching the target $80\%$ theoretical coverage).
   - Median prediction interval width: **$277.2\text{ minutes}$ ($4.62\text{ hours}$)**.
2. **Probability Calibration (Validation Prefit)**:
   - Uncalibrated LightGBM had high calibration error ($\text{ECE} = 0.2899$, $\text{Brier} = 0.1906$) due to cost-sensitive reweighting.
   - **Isotonic Calibration** (fit on Validation set) reduced Test Expected Calibration Error to **$\text{ECE} = 0.0292$** and **$\text{Brier} = 0.0860$** (a **$10\times$ error reduction**).
3. **Data-Driven Advisory Buffer**:
   - Derived empirically from test positive residual overrun percentiles ($P_{80} = +174.5\text{m}$, $P_{90} = +258.9\text{m}$) rather than arbitrary heuristics.

---

## 🌐 5. Multi-Split Generalization Benchmark

| Split Strategy | Evaluation Scenario | Test MAE (min) | Test $R^2$ | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| **Temporal (70/15/15)** | **Chronological Future Generalization (Production)** | **93.34** | **0.2530** | **0.7886** | **0.3461** |
| **Courier-Grouped Holdout** | **Cold-Start Generalization to Unseen Couriers** | 87.41 | -0.1117 | 0.7251 | 0.2185 |
| **Random (80/20)** | **Uniform Interpolation Baseline (Point-in-Time)** | 84.99 | 0.3128 | 0.8118 | 0.4012 |

---

## 🏛️ 6. System Architecture & Performance Latency

```text
 ┌──────────────────────────────────────────────────────────────┐
 │                      Client Layer                            │
 │  ┌────────────────────────────┐  ┌────────────────────────┐  │
 │  │ Streamlit App (:8501)      │  │ Logistics ERP / Fleet  │  │
 │  └─────────────┬──────────────┘  └───────────┬────────────┘  │
 └────────────────┼─────────────────────────────┼───────────────┘
                  │                             │
                  ▼                             ▼
 ┌──────────────────────────────────────────────────────────────┐
 │             FastAPI Microservice Engine (:8000)              │
 │  • POST /predict/dispatch-evaluation                         │
 │  • POST /predict/raw-gps                                     │
 │  • POST /predict/batch                                       │
 │  • GET  /health & /models/rules                              │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                 Calibrated ML Models & Engine                │
 │  • LightGBM Point ETA Regressor                              │
 │  • Quantile Regressors (P10, P50, P90 Intervals)             │
 │  • Isotonic-Calibrated High-Delay Risk Classifier            │
 │  • Empirical Sequence Advisory Engine                        │
 └──────────────────────────────────────────────────────────────┘
```

### Production Latency Benchmark (Measured across 300 requests):
- **Single Request Latency**: Mean **$24.96\text{ ms}$**, Median **$24.99\text{ ms}$**, $P_{95}$ **$28.66\text{ ms}$**, $P_{99}$ **$31.30\text{ ms}$** (Throughput: **$40.1\text{ QPS}$**).
- **Batch Evaluation (50 orders)**: **$20.97\text{ ms}$ per order**.
