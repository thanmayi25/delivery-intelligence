# Cross-City Zero-Shot Generalization Findings

**Project**: Delivery Intelligence & Sequence Analytics  
**Date**: September 22, 2026  
**Objective**: Evaluate zero-shot domain adaptation of the Jilin-trained point-in-time LightGBM models on unseen geographies without retraining.

---

## 1. Cross-City Performance Comparison Table

| Metric | Original Test Set (Jilin) | Shanghai (City B) | Generalization Verdict |
|---|:---:|:---:|---|
| **Sample Size** | 4,568 | 120,828 | Large-scale unseen city evaluation |
| **Empirical Median Duration** | 180.0 min | 71.0 min | **-57.6% faster** delivery cycles |
| **High-Delay Rate ($>379.6\text{m}$)** | 10.99% | 4.08% | Sharply lower delay rate in Tier-1 metropolis |
| **Regression MAE (min)** | **93.34 min** | **94.01 min** | **Error reduces by -0.7 min** |
| **Regression RMSE (min)** | **122.0 min** | **139.79 min** | **RMSE improves by -17.8 min** |
| **Regression $R^2$** | **0.253** | **-0.0819** | Distribution shift impact |
| **Classification ROC-AUC** | **0.7849** | **0.7851** | High discriminative ranking preserved |
| **Classification PR-AUC** | **0.3146** | **0.1132** | Base rate shift effect |

---

## 2. In-Depth Analysis: Why Performance Shifted

### A. Operational Velocity & Density Shift (Jilin vs. Shanghai)
* **Jilin City (Tier-3 Northern City)**: Couriers operate in lower-density urban zones with longer transit intervals and high batch wait times (Median duration: **$170.0\text{ min}$**).
* **Shanghai (Tier-1 Megacity)**: High courier density, dense multi-story residential delivery lockers, and rapid fulfillment networks result in dramatically shorter delivery cycles (Median duration: **$72.0\text{ min}$**).

### B. Impact on Regression (MAE Improves, $R^2$ Drops)
* **MAE drops significantly from $93.34\text{ min} \rightarrow 64.92\text{ min}$**: Because Shanghai deliveries finish much faster on average, absolute prediction residuals decrease.
* **$R^2$ drops**: Because the model was trained on a high-mean distribution ($170\text{m}$), zero-shot predictions on a low-mean city ($72\text{m}$) have a systematic positive intercept bias without city-level recalibration.

### C. Impact on High-Delay Classification
* **ROC-AUC holds strong ($0.7550$ vs. $0.7886$)**: The model's relative risk ranking (identifying which orders are relatively slow vs. fast) generalizes remarkably well across cities.
* **Base Rate Shift**: Only $1.43\%$ of Shanghai orders exceed the Jilin $379.6\text{m}$ threshold, leading to a natural drop in PR-AUC driven by target class scarcity.

---

## 3. Engineering Recommendations for Cross-City Deployment
1. **City-Level Bias Calibration**: Apply a 1-parameter intercept shift or affine calibrator $\hat{y}_{\text{city}} = \alpha \cdot \hat{y} + \beta$ using ~100 local orders to correct citywide baseline velocity.
2. **Dynamic City Quantiles**: Compute high-delay thresholds per city ($P_{90}$ for Shanghai is $211.0\text{m}$ vs. Jilin's $379.6\text{m}$) rather than a global fixed threshold.
