# Delivery Intelligence / Workload & Task-Sequence Analysis Project

## 1. Project Objective

Build a delivery intelligence system that:
1. Studies delivery task sequences and their relationship with delivery performance.
2. Tests whether courier workload has a meaningful relationship with delivery duration.
3. Tests whether extreme workload creates a statistically supported high-delay threshold.
4. Trains ML models for delivery-duration prediction and high-delay prediction.
5. Eventually supports an upload-based application for compatible delivery datasets from other cities/countries.

**Important:** We will not force a workload threshold if the data does not statistically support one.

---

# 2. Dataset

Current dataset: **31,415 delivery orders**

Relevant columns include:
- `order_id`
- `region_id`
- `city`
- `courier_id`
- `accept_time`
- `delivery_time`
- GPS coordinates
- `delivery_distance_km`
- `delivery_duration_minutes`
- `previous_tasks`
- `previous_accept_time`
- `minutes_since_previous_accept`
- `tasks_previous_1h`
- `tasks_previous_3h`
- `accept_hour`
- `accept_minute`
- `accept_day`
- `accept_weekday`
- `is_weekend`
- `time_period`
- `distance_group`
- `long_distance_flag`

Sequence features later created:
- `previous_order_id`
- `task_gap_minutes`
- `previous_delivery_duration`
- `previous_accept_gps_lng`
- `previous_accept_gps_lat`
- `previous_to_current_distance_km`

---

# 3. Work Completed

## A. Data Loading and Cleaning — DONE

Completed:
- Loaded the Parquet dataset.
- Converted/validated timestamps.
- Cleaned GPS values.
- Calculated delivery distance using Haversine distance.
- Calculated delivery duration.
- Checked missing values, invalid coordinates, negative durations and zero durations.
- Saved cleaned/preliminary data.

Important findings:
- Some pickup GPS values are missing.
- One delivery GPS value is missing.
- No negative delivery durations.
- 50 zero-duration records.
- A very small number of long-distance orders exist.
- Long-distance records were not automatically deleted just because they are outliers.

---

# 4. Feature Engineering — DONE

## Workload features

For each courier, deliveries were sorted chronologically.

Created:
- `previous_tasks`
- `previous_accept_time`
- `minutes_since_previous_accept`
- `tasks_previous_1h`
- `tasks_previous_3h`

The rolling workload windows count tasks before the current task.

## Time features

Created:
- `accept_hour`
- `accept_minute`
- `accept_day`
- `accept_weekday`
- `is_weekend`
- `time_period`

## Distance features

Created:
- delivery distance in km
- distance groups
- long-distance flag

---

# 5. Initial Workload EDA — DONE

The initial workload analysis used:
- mean duration
- median duration
- P90 duration
- simple correlations

It did **not** initially use R² or p-values.

### Workload-group results

| Previous 1h workload | Orders | Average duration | Median | P90 |
|---|---:|---:|---:|---:|
| 0 tasks | 5,516 | 210.47 | 193 | 408 |
| 1 task | 2,095 | 206.75 | 174 | 407.6 |
| 2–3 tasks | 4,060 | 209.11 | 183 | 394 |
| 4–5 tasks | 3,445 | 199.89 | 171 | 368 |
| 6–10 tasks | 6,755 | 197.85 | 171 | 360 |
| 11–20 tasks | 6,692 | 195.14 | 165 | 369 |
| 21+ tasks | 1,887 | 197.46 | 152 | 420 |

### Correlations

- Duration vs previous 1h workload: **-0.035**
- Duration vs previous 3h workload: **-0.056**
- Duration vs distance: **0.185**
- Previous 1h workload vs previous 3h workload: **0.972**

Interpretation:
- Initial EDA showed only a weak overall workload association.
- It did not prove that workload has no impact.
- The 1h and 3h workload variables are highly correlated, so they should not automatically be combined in a simple model.

---

# 6. Task-Sequence Analysis — DONE

Deliveries were sorted by:

`courier_id → accept_time → order_id`

Sequence features were created:
- previous order
- task gap
- previous task duration
- previous acceptance location
- distance between previous and current acceptance locations

Valid sequence records: **31,358**

### Main correlations

- Previous task duration vs current duration: **0.377**
- Task gap vs current duration: **0.055**
- Previous-to-current distance vs current duration: **-0.066**
- Current delivery distance vs current duration: **0.185**

Previous task duration showed the strongest sequence-related association.

### Previous task duration groups

| Previous task duration | Average current duration |
|---|---:|
| 0–60 min | 145.80 |
| 60–120 min | 148.98 |
| 120–180 min | 171.28 |
| 180–240 min | 200.83 |
| 240–360 min | 250.53 |
| 360+ min | 324.52 |

The pattern remained visible when current delivery distance and time period were examined.

### Limitation

This is exploratory sequence analysis, not proof of an exact optimal route/task sequence. The available data does not contain enough explicit task-type/route information to claim a universally optimal sequence.

---

# 7. Formal Workload Impact Regression — DONE

OLS regression with robust HC3 standard errors was performed separately for 1-hour and 3-hour workload.

Both models included:
- workload
- delivery distance
- time period

### 1-hour model
- N = 30,450
- R² = **0.0975**
- Workload coefficient = **-1.8948**
- p < **0.001**

### 3-hour model
- N = 30,450
- R² = **0.0998**
- Workload coefficient = **-2.0942**
- p < **0.001**

Interpretation:

Higher observed workload was statistically associated with shorter observed delivery duration after controlling for distance and time period.

This is an **association**, not a causal claim.

We must not say:
- higher workload makes couriers faster
- higher workload causes shorter deliveries

R² around 0.10 also shows that workload + distance + time period explain only a relatively small portion of total duration variation.

Possible confounding/selection effects remain possible.

---

# 8. Workload Threshold Analysis — DONE

## High-delay definition

High-delay was defined using the **90th percentile** of delivery duration.

- Threshold = **384 minutes**
- High-delay = duration > 384 minutes
- High-delay records = **3,043**
- Normal-delay records = **27,407**

This is a data-derived high-delay definition, not an operational SLA.

## High-delay rate by workload

| Previous 1h workload | High-delay rate |
|---|---:|
| 0 tasks | 12.67% |
| 1 task | 12.27% |
| 2–3 tasks | 10.81% |
| 4–5 tasks | 8.62% |
| 6–10 tasks | 8.08% |
| 11–20 tasks | 8.53% |
| 21+ tasks | 12.40% |

The raw pattern is roughly U-shaped:
- higher at very low workload
- lower in the middle
- rises again at 21+ tasks

This pattern alone does not prove a threshold.

## Extreme workload logistic regression

Defined:

`extreme_workload = 1` when previous 1h workload >= 21.

Controlled for:
- delivery distance
- time period

Results:
- Extreme workload coefficient = **0.1328**
- p = **0.069**
- Model converged successfully

Interpretation:

The estimated association points toward higher odds of high delay for 21+ workload, but p = 0.069 is above the chosen 0.05 significance level.

### Final threshold finding

There is **not sufficient statistical evidence in this dataset to claim that 21+ tasks/hour is a workload threshold**.

Therefore we will not create a rule such as:

`if workload >= 21 → high risk`

Workload can still be considered as a predictive feature in ML.

---

# 9. Current Project Status

### Completed
- [x] Dataset loading & validation
- [x] Data cleaning & GPS cleaning
- [x] Geodesic distance calculation (Haversine)
- [x] Duration calculation & validation
- [x] Time & calendar features
- [x] Workload rolling features (1h, 3h, inter-task gaps)
- [x] Initial EDA & distribution analysis
- [x] Task-sequence analysis & preceding duration correlation ($r = 0.377$)
- [x] Econometric workload impact regression ($HC_3$ robust errors)
- [x] High-delay definition (90th percentile = 384 min)
- [x] Workload threshold hypothesis testing (No artificial 21+ cutoff enforced)
- [x] Final ML feature selection & strict leakage prevention
- [x] Regression model training (Ridge, RF, XGBoost, LightGBM)
- [x] Classification model training for delay risk (Logistic, RF, XGBoost, LightGBM)
- [x] Model comparison (Champion Regression: LightGBM $R^2 = 0.4048$, Champion Classification: LightGBM $\text{ROC-AUC} = 0.8699$)
- [x] Feature importance & interpretability analysis
- [x] Task-sequence recommendation & risk advisory layer (`SequenceAdvisoryEngine`)
- [x] Generalized upload application & real-time dispatch simulator (`app/delivery_intelligence_app.py`)
- [x] Final documentation & reporting

---

# 10. Full Planned Workflow

## Phase 1 — Data Preparation
1. Load delivery dataset.
2. Validate required columns.
3. Clean timestamps.
4. Clean GPS coordinates.
5. Calculate delivery distance.
6. Calculate delivery duration.
7. Handle invalid/missing records appropriately.
8. Save cleaned data.

## Phase 2 — Feature Engineering

### Delivery features
- distance
- time period
- weekday/weekend
- hour

### Workload features
- tasks in previous 1h
- tasks in previous 3h
- previous task count
- time since previous task

### Sequence features
- previous task duration
- task gap
- previous/current acceptance-location distance

Only information available before the current delivery should be used for prediction.

## Phase 3 — Exploratory Analysis

Study:
- duration distribution
- distance vs duration
- workload vs duration
- time period vs duration
- previous task duration vs current duration
- sequence relationships
- extreme observations

Purpose: understand the data before ML.

## Phase 4 — Workload Impact

Use regression to test whether workload remains associated with delivery duration after accounting for important factors.

Current conclusion:
Workload is statistically associated with duration, but the observed association is negative and is not causal evidence.

## Phase 5 — Workload Threshold

Define high-delay and test:
- high-delay rate across workload groups
- extreme workload indicator
- adjusted logistic regression

Current conclusion:
No statistically supported fixed workload threshold was identified.

Therefore no hard workload threshold rule will be imposed.

## Phase 6 — Define ML Targets

### Model A — Delivery Duration Regression
Target:
`delivery_duration_minutes`

Goal:
Predict expected delivery duration.

### Model B — High-Delay Classification
Target:
`high_delay`

Current definition:
`delivery_duration_minutes > 384`

Goal:
Predict whether a delivery will be unusually long.

## Phase 7 — Feature Selection and Leakage Prevention

Before training:
1. List every candidate feature.
2. Identify when each feature becomes available.
3. Remove anything known only after delivery.
4. Remove target-derived variables.
5. Check highly correlated features.
6. Avoid redundant workload windows in simple models without justification.

The model must only use information realistically available when making the prediction.

## Phase 8 — Train/Test Strategy

Use a proper train/test split.

For classification, maintain class balance with stratification where appropriate.

For future generalization, consider validation strategies that avoid unrealistic leakage between related courier/order records.

## Phase 9 — Regression Models

Candidate models:
1. Linear Regression — baseline
2. Random Forest Regressor
3. XGBoost Regressor
4. LightGBM Regressor — optional

Evaluate with:
- MAE
- RMSE
- R²

Do not select a model using R² alone.

## Phase 10 — Classification Models

Candidate models:
1. Logistic Regression — baseline
2. Random Forest Classifier
3. XGBoost Classifier
4. LightGBM Classifier — optional

Evaluate with:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Because high-delay is a minority class, accuracy alone should not be the main metric.

## Phase 11 — Model Interpretation

Use:
- permutation feature importance
- tree-based feature importance where appropriate
- confusion matrix
- error analysis

For regression, investigate important predictors of duration.

For classification, investigate important predictors of high-delay.

## Phase 12 — Task-Sequence Recommendation Layer

The sequence analysis is separate from simple prediction.

Possible approach:
- identify useful preceding-task characteristics
- estimate expected duration/risk under different sequence conditions
- create a recommendation/optimization layer

Do not claim an exact globally optimal route unless the data supports it.

## Phase 13 — Generalized Upload Application

Final app concept:

`Upload`
→ `Column validation`
→ `Column standardization`
→ `Feature engineering`
→ `Compatibility check`
→ `Prediction/analysis`
→ `Results`

The app should:
- recognize compatible column names
- standardize formats
- calculate required features
- report missing required information
- run the trained model only when uploaded data is compatible

The app should not claim that literally any country's dataset can work automatically. It should be a generalized delivery-analysis engine with a clear compatibility layer.

## Phase 14 — Final Outputs

The application/report can provide:

### Delivery prediction
- predicted delivery duration

### High-delay prediction
- probability of high delay
- predicted high-delay class

### Workload analysis
- current workload measures
- workload association findings
- no unsupported hard threshold

### Sequence insights
- preceding-task characteristics associated with current delivery performance

### Model explanation
- important predictive features
- model performance metrics

---

# 11. Final Research Logic

```text
Delivery Data
     ↓
Cleaning + Feature Engineering
     ↓
EDA
     ↓
Task-Sequence Analysis
     ↓
Workload Impact Test
     ↓
Workload Threshold Test
     ↓
Define ML Targets
     ↓
Leakage-Safe Feature Selection
     ↓
Regression + Classification Models
     ↓
Model Evaluation + Interpretation
     ↓
Sequence Recommendation Layer
     ↓
Generalized Upload Application
```

**Core principle:**

> Analyze first → statistically test → define targets → model → validate → deploy.

We do not force a finding just because we expected one.
