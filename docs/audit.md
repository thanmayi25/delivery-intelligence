# Phase 0 Audit Report: Delivery Intelligence & Sequence Analytics

**Audit Date**: September 20, 2026  
**Auditor**: Senior Machine Learning Engineer  
**Status**: Completed — Phase 0 Acceptance Criteria Satisfied  

---

## 1. Dataset Facts & Domain Characterization

### 1.1 Empirical Dataset Properties
An exhaustive audit of the primary training dataset ([`data/raw/delivery/delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/data/raw/delivery/delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet)) reveals the following verified facts:

| Dimension | Measured Value | Analysis & Context |
|---|---|---|
| **Raw Row Count** | **31,415** | 30,450 clean rows after dropping missing GPS / zero distance records |
| **Unique Orders** | **31,415** | Every row represents a distinct order assignment |
| **Unique Couriers** | **57** | Extremely small courier pool (mean: 551.1 orders/courier, median: 489 orders) |
| **Unique Regions** | **4** | Regions 31, 53, 129, 147 within Jilin City |
| **Calendar Span** | **163 days** | May 12 to October 28 (recorded in `ds` codes 512 to 1028) |
| **Daily Volume** | Mean: **192.7 orders/day** | Range: 1 to 434 orders/day across all 57 couriers |

---

### 1.2 Delivery Duration Distribution

Delivery duration is computed as $(\text{delivery\_time} - \text{accept\_time})$ in minutes.

| Metric | Duration (Minutes) | Duration (Hours) | Percentage of Dataset |
|---|---|---|---|
| **Minimum** | 0.0 min | 0.00 h | 50 records (0.16%) |
| **25th Percentile ($P_{25}$)** | 99.0 min | 1.65 h | 25.0% |
| **Median ($P_{50}$)** | **175.0 min** | **2.92 h** | 50.0% |
| **Mean** | **203.5 min** | **3.39 h** | — |
| **75th Percentile ($P_{75}$)** | **276.0 min** | **4.60 h** | 75.0% |
| **90th Percentile ($P_{90}$)** | **387.0 min** (384m clean) | **6.45 h** | 90.0% |
| **95th Percentile ($P_{95}$)** | 469.0 min | 7.82 h | 95.0% |
| **99th Percentile ($P_{99}$)** | 660.9 min | 11.01 h | 99.0% |
| **Maximum** | 3,573.0 min | 59.55 h | Extreme outlier |

```
Distribution of Delivery Duration:
< 1 hour:    12.06%  (3,789 orders)
1 – 3 hours: 39.67% (12,463 orders)
3 – 6 hours: 35.75% (11,229 orders)
> 6 hours:   12.52%  (3,934 orders)
```

---

### 1.3 True Operational Domain: Express Last-Mile Parcel Delivery vs. On-Demand Food

> [!CAUTION]
> **Domain Framing Misalignment**: The current codebase, README, and UI repeatedly describe this system as *"on-demand urban logistics / instant food delivery"*.
> 
> **Evidence of True Domain**:
> 1. **Median duration is 2.92 hours** (almost 3 hours), and 87.94% of deliveries take $> 1\text{ hour}$. On-demand food delivery operates on a 25–45 minute horizon.
> 2. **Batch Wave Dispatch**: 55.30% of all orders are assigned at the *exact same minute* as other orders to the same courier.
> 3. **Courier Scale**: Only 57 couriers handle 31,415 packages over 5.5 months in designated delivery zones (AOIs).
> 
> **Correct Characterization**: This dataset represents **last-mile e-commerce parcel / express logistics (e.g., Cainiao / SF Express / JD style wave delivery)**, where couriers receive wave dispatches (10–30 parcels in morning and afternoon waves) and execute distribution runs across multi-hour shifts.

---

## 2. Explanation of the 30k vs. 931k Row Scale Discrepancy

* **Repository Model Training Dataset ($N = 31,415$)**:
  - Located at [`data/raw/delivery/delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/data/raw/delivery/delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet).
  - Represents the **Jilin (`JL`) city partition** of the open Alibaba/Cainiao benchmark dataset.
* **Batch Analyzer Scale ($N = 931,351$)**:
  - Represents the **Chongqing (`CQ`) city partition** from the same dataset family, which was ingested externally via the Streamlit file uploader during batch stress testing.
  - The models in `/models/` were trained on Jilin ($30\text{k}$) and tested for generalizability/inference speed on Chongqing ($931\text{k}$).

---

## 3. Feature-by-Feature Point-in-Time Leakage Audit

A strict point-in-time check was conducted for every feature in [`scripts/03_ml_preprocessing.py`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/scripts/03_ml_preprocessing.py):

| Feature Name | Calculation Method | Known at Dispatch Time? | Leakage & Integrity Assessment |
|---|---|:---:|---|
| `delivery_distance_km` | Haversine distance between `accept_gps` and `delivery_gps` | **YES** | Legitimate. Geodesic distance is known upon assignment. |
| `tasks_previous_1h` | `rolling("1h", closed="left").count()` on `accept_time` | **YES** | Legitimate. Counts orders accepted in the prior 60 minutes strictly before current `accept_time`. |
| `tasks_previous_3h` | `rolling("3h", closed="left").count()` on `accept_time` | **YES** | Legitimate. However, collinearly correlated with 1h workload ($r = 0.972$). |
| `previous_tasks` | `groupby("courier_id").cumcount()` | **LEAKAGE / FLAWED** | **Flawed**: Cumulates over all 163 calendar days without resetting daily. A courier on Day 150 gets `previous_tasks = 850`, leaking absolute chronological position. Must reset per shift/day. |
| `task_gap_minutes` | `accept_time[i] - accept_time[i-1]` | **YES** | Legitimate. Measures time since prior task assignment. |
| `previous_delivery_duration` | `groupby("courier_id")["delivery_duration_minutes"].shift(1)` | 🚨 **CRITICAL LEAKAGE** | **Severe Target Leakage**: See Section 3.1 below. |
| `previous_to_current_distance_km` | Haversine between `accept_gps[i-1]` and `accept_gps[i]` | **YES** | Legitimate. Prior pickup coordinates are known. |
| `is_first_task_of_courier` | `previous_tasks == 0` | **LEAKAGE / FLAWED** | Only flags the first order on Day 1 (May 12) per courier (57 rows total). Does not identify daily shift starts. |
| `accept_hour`, `accept_minute`, `accept_weekday`, `is_weekend`, `time_period` | Extracted from `accept_time` | **YES** | Legitimate local temporal features. |

---

### 3.1 Critical Finding: The `previous_delivery_duration` Target Leakage Mechanism

```
Courier 2289 on Date 529 (Wave Dispatch at 08:30):
  Order 1: Dispatched 08:30:00 ──► Delivered at 13:45:00 (Duration = 315 min)
  Order 2: Dispatched 08:30:00 ──► Delivered at 14:10:00 (Duration = 340 min)
  
  At Order 2 Dispatch Time (08:30:00):
  - Order 1 has NOT happened yet (it finishes 5h 15m in the FUTURE).
  - Code used `shift(1)`, assigning 315 min as a feature to Order 2 at 08:30:00!
```

#### Quantitative Audit on Real Data:
- **Total Sequential Task Pairs**: $31,358$
- **Pairs where previous task was UNFINISHED at current dispatch time**: **$28,610$ ($91.24\%$)**
- **Pairs where previous task was FINISHED before current dispatch time**: **$2,748$ ($8.76\%$)**
- **Orders assigned at the EXACT SAME MINUTE (Batch Wave Dispatch)**: **$17,341$ ($55.30\%$)**

> [!IMPORTANT]
> **Why the $r = 0.377$ correlation and "8x delay risk" occurred**:  
> Because $91.24\%$ of previous tasks were unfinished at dispatch time, `previous_delivery_duration` leaked the *future duration of concurrent batch orders from the same morning shift*. When a courier had a heavy morning shift that took until late afternoon, all orders in that batch had long future durations, creating an artificial correlation.
>
> **Required Fix for Phase 1**: The feature must strictly use the duration of the **most recently COMPLETED delivery as of the current task's dispatch timestamp** (or elapsed time of active in-flight tasks).

---

## 4. Train / Test Split Audit

### 4.1 Current Implementation
* **Code Location**: [`scripts/03_ml_preprocessing.py: Lines 144–150`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/scripts/03_ml_preprocessing.py#L144-L150)
* **Method**:
  ```python
  train_df, test_df = train_test_split(
      full_dataset, test_size=0.20, random_state=42, stratify=full_dataset["high_delay"]
  )
  ```

### 4.2 Critical Issues Identified

1. **Temporal Data Leakage**:
   - Random sampling selects records uniformly across the 163-day timeline (May to October).
   - Future data (e.g. October) is in the training set while past data (e.g. June) is in the test set. Real deployment requires predicting the *future*.
2. **Within-Courier Batch Contamination**:
   - Couriers receive batches of 15–25 parcels on a single day.
   - Random splitting places 12 parcels from a specific courier's Tuesday shift into the training set and 3 parcels into the test set. The model memorizes courier-day specific route conditions.
3. **Target Threshold Leakage**:
   - The $P_{90}$ threshold ($384\text{ min}$) was computed on the *entire dataset* prior to splitting.
   - The threshold must be calculated strictly on the training partition and applied to validation/test sets.

---

## 5. Audit of Unsubstantiated or Overstated Claims

The following claims across documentation and code violate scientific conservatism and must be corrected in Phase 1:

| Artifact | Location | Overstated / Unsubstantiated Claim | Empirical Reality & Required Correction |
|---|---|---|---|
| [`README.md`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/README.md) & [`PROJECT_OVERVIEW.md`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/PROJECT_OVERVIEW.md) | Section 2 | *"Finding 1: Task-Sequence Ripple Effects are Dominant ($r = 0.377$)"* and *"8x risk increase"* | **Leakage Artifact**: 91.24% of prior tasks were unfinished at dispatch time. Must re-evaluate under point-in-time completed task duration. |
| [`README.md`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/README.md) & [`PROJECT_OVERVIEW.md`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/PROJECT_OVERVIEW.md) | Section 2 | *"Rejection of Arbitrary Thresholds / Proven no threshold exists"* | **Overstated Claim**: Logistic regression yielded $p = 0.069$. In statistical hypothesis testing, failing to reject $H_0$ at $\alpha=0.05$ is **not proof that no effect exists**. Must report effect size, CI, and state "no statistically significant evidence was observed." |
| [`README.md`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/README.md) & [`PROJECT_OVERVIEW.md`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/PROJECT_OVERVIEW.md) | Section 4 | Hardcoded metric tables (MAE 74.59 min, $R^2 = 0.4048$, ROC-AUC 0.8699) | **Hardcoded**: Violates Ground Rule 3. All metrics must be dynamically loaded from `/results/` JSON/CSV artifacts. |
| [`app/delivery_intelligence_app.py`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/app/delivery_intelligence_app.py) | Line 32 | Causal terms: *"Cascading Delay Propagation"*, *"Ripple Effect"* | **Unjustified Causal Language**: Observational sequence correlation does not establish causality. Must use cautious phrasing (*"Empirical Sequence Association"*). |
| [`app/delivery_intelligence_app.py`](file:///c:/Users/thanmayi/OneDrive/Desktop/delivery%20intelligence/app/delivery_intelligence_app.py) | Line 486 | Fixed $+45\text{m}$ buffer recommendation | **Arbitrary Rule**: The $+45\text{m}$ buffer was hardcoded rather than derived empirically from test residual distributions. |

---

## 6. Audit Action Plan for Phase 1

1. **Temporal & Grouped Split Implementation**:
   - Establish Chronological Train (70%, May 12 – Aug 28), Validation (15%, Aug 29 – Sep 15), and Test (15%, Sep 16 – Oct 28).
   - Implement Courier GroupKFold holdout as secondary validation.
   - Calculate $P_{90}$ high-delay target strictly on training split.
2. **Point-in-Time Sequence Feature Engineering**:
   - Replace future `shift(1)` with true point-in-time lookups: `duration_of_most_recently_COMPLETED_task_as_of_dispatch_time`.
   - Implement automated leakage unit test.
3. **Probability & Uncertainty Calibration**:
   - Fit isotonic/sigmoid calibration on validation split.
   - Implement LightGBM quantile regression ($P_{10}, P_{50}, P_{90}$) and empirical residual buffer calculation.
4. **Dynamic Reporting Pipeline**:
   - Save all split, baseline, ablation, and calibration outputs to `/results/`.
   - Bind README, docs, and UI dynamically to `/results/`.
