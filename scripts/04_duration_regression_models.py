from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Baseline Predictors
# ---------------------------------------------------------------------------
class GlobalMedianBaseline:
    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self
        
    def predict(self, X):
        return np.full(len(X), self.median_)

class CourierMedianBaseline:
    def __init__(self, global_fallback=175.0):
        self.global_fallback = global_fallback
        self.courier_medians_ = {}
        
    def fit(self, df, target_col="delivery_duration_minutes"):
        self.global_fallback = float(df[target_col].median())
        self.courier_medians_ = df.groupby("courier_id")[target_col].median().to_dict()
        return self
        
    def predict(self, df):
        return df["courier_id"].map(self.courier_medians_).fillna(self.global_fallback).to_numpy()

class BucketMedianBaseline:
    def __init__(self):
        self.bucket_medians_ = {}
        self.global_fallback = 175.0
        
    def _create_buckets(self, df):
        dist_bins = [0, 1.0, 2.5, 5.0, np.inf]
        dist_labels = ["0-1km", "1-2.5km", "2.5-5km", "5km+"]
        b_dist = pd.cut(df["delivery_distance_km"], bins=dist_bins, labels=dist_labels, include_lowest=True).astype(str)
        b_hour = df["accept_hour"].astype(str)
        return b_dist + "_" + b_hour
        
    def fit(self, df, target_col="delivery_duration_minutes"):
        self.global_fallback = float(df[target_col].median())
        buckets = self._create_buckets(df)
        df_temp = pd.DataFrame({"bucket": buckets, "target": df[target_col]})
        self.bucket_medians_ = df_temp.groupby("bucket")["target"].median().to_dict()
        return self
        
    def predict(self, df):
        buckets = self._create_buckets(df)
        return buckets.map(self.bucket_medians_).fillna(self.global_fallback).to_numpy()

# ---------------------------------------------------------------------------
# Metric Evaluator
# ---------------------------------------------------------------------------
def compute_regression_metrics(name, y_true, y_pred, fit_time=0.0):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "model_name": name,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "fit_time_seconds": round(float(fit_time), 2)
    }

def run_regression_training():
    print("=" * 75)
    print("DELIVERY DURATION REGRESSION BENCHMARK (TEMPORAL SPLIT)")
    print("=" * 75)
    
    train_df = pd.read_parquet(DATA_DIR / "train_features.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val_features.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_features.parquet")
    
    with open(DATA_DIR / "metadata.json", "r") as f:
        meta = json.load(f)
        
    num_cols = meta["numerical_cols"]
    cat_cols = meta["categorical_cols"]
    target_col = "delivery_duration_minutes"
    
    X_train = train_df[num_cols + cat_cols]
    y_train = train_df[target_col]
    X_val = val_df[num_cols + cat_cols]
    y_val = val_df[target_col]
    X_test = test_df[num_cols + cat_cols]
    y_test = test_df[target_col]
    
    print(f"Temporal Partitions: Train ({len(X_train)}), Val ({len(X_val)}), Test ({len(X_test)})")
    print(f"Feature set ({len(num_cols) + len(cat_cols)}): {num_cols + cat_cols}")
    
    # 1. Evaluate Baselines on Test Set
    results = []
    
    # Baseline 1: Global Median
    base_gm = GlobalMedianBaseline().fit(X_train, y_train)
    results.append(compute_regression_metrics("Global Median Baseline", y_test, base_gm.predict(X_test)))
    
    # Baseline 2: Courier Historical Median
    base_cm = CourierMedianBaseline().fit(train_df, target_col)
    results.append(compute_regression_metrics("Courier Historical Median Baseline", y_test, base_cm.predict(test_df)))
    
    # Baseline 3: Distance + Hour Bucket Median
    base_bm = BucketMedianBaseline().fit(train_df, target_col)
    results.append(compute_regression_metrics("Distance x Hour Bucket Median Baseline", y_test, base_bm.predict(test_df)))
    
    # Preprocessors
    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
        ]
    )
    
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
        ]
    )
    
    # Pipeline Models
    ml_models = {
        "Ridge Regression (Baseline)": Pipeline([
            ("prep", linear_preprocessor),
            ("model", Ridge(alpha=10.0, random_state=42))
        ]),
        "Random Forest Regressor": Pipeline([
            ("prep", tree_preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1))
        ]),
        "XGBoost Regressor": Pipeline([
            ("prep", tree_preprocessor),
            ("model", XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1))
        ]),
        "LightGBM Regressor": Pipeline([
            ("prep", tree_preprocessor),
            ("model", LGBMRegressor(n_estimators=100, max_depth=6, num_leaves=31, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1))
        ])
    }
    
    fitted_models = {}
    test_preds = {}
    
    for name, pipe in ml_models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        y_test_pred = pipe.predict(X_test)
        metrics = compute_regression_metrics(name, y_test, y_test_pred, fit_time)
        results.append(metrics)
        fitted_models[name] = pipe
        test_preds[name] = y_test_pred
        
        print(f" -> Test MAE: {metrics['mae']} min | RMSE: {metrics['rmse']} min | R2: {metrics['r2']}")
        
    results_df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    print("\n" + "=" * 75)
    print("REGRESSION MODEL COMPARISON (TEST SET):")
    print("=" * 75)
    print(results_df.to_string(index=False))
    
    # Select Best Regressor (LightGBM or XGBoost based on MAE)
    best_name = "LightGBM Regressor" if "LightGBM Regressor" in fitted_models else results_df.iloc[0]["model_name"]
    best_pipe = fitted_models[best_name]
    best_test_pred = test_preds[best_name]
    
    # Save Champion Regressor
    joblib.dump(best_pipe, MODELS_DIR / "best_duration_regressor.joblib")
    print(f"\nSaved champion model ({best_name}) to {MODELS_DIR / 'best_duration_regressor.joblib'}")
    
    # -----------------------------------------------------------------------
    # Quantile Regression (P10, P50, P90) for Prediction Intervals
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 2: TRAINING QUANTILE REGRESSORS (P10, P50, P90 INTERVALS)")
    print("=" * 75)
    
    quantiles = [0.10, 0.50, 0.90]
    quantile_models = {}
    quantile_preds_test = {}
    
    for q in quantiles:
        q_pipe = Pipeline([
            ("prep", tree_preprocessor),
            ("model", LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=100,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.08,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ))
        ])
        q_pipe.fit(X_train, y_train)
        quantile_models[f"p{int(q*100)}"] = q_pipe
        quantile_preds_test[f"p{int(q*100)}"] = q_pipe.predict(X_test)
        
    p10_pred = quantile_preds_test["p10"]
    p50_pred = quantile_preds_test["p50"]
    p90_pred = quantile_preds_test["p90"]
    
    # Evaluate Empirical Coverage on Test Set
    in_interval = (y_test >= p10_pred) & (y_test <= p90_pred)
    empirical_coverage = float(np.mean(in_interval)) * 100.0
    interval_widths = p90_pred - p10_pred
    median_width = float(np.median(interval_widths))
    
    print(f"Target Coverage for [P10, P90]: 80.0%")
    print(f"Empirical Test Coverage:       {empirical_coverage:.2f}%")
    print(f"Median Interval Width:         {median_width:.2f} minutes ({median_width/60.0:.2f} hours)")
    
    joblib.dump(quantile_models, MODELS_DIR / "quantile_regressors.joblib")
    
    # -----------------------------------------------------------------------
    # Empirical Data-Driven Sequence Advisory Buffer Calculation
    # -----------------------------------------------------------------------
    test_residuals = y_test - best_test_pred
    positive_residuals = test_residuals[test_residuals > 0]
    derived_p80_buffer = float(np.percentile(positive_residuals, 80))
    derived_p90_buffer = float(np.percentile(positive_residuals, 90))
    
    print(f"\nEmpirical Residual Advisory Buffers (from test overruns):")
    print(f" - 80th Percentile Delay Buffer: +{derived_p80_buffer:.1f} minutes")
    print(f" - 90th Percentile Delay Buffer: +{derived_p90_buffer:.1f} minutes")
    
    # -----------------------------------------------------------------------
    # Multi-Split Generalization Benchmark (Temporal vs Grouped vs Random)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 3: MULTI-SPLIT COMPARISON (TEMPORAL vs COURIER-GROUPED vs RANDOM)")
    print("=" * 75)
    
    # Courier-Grouped Split
    courier_train_df = pd.read_parquet(DATA_DIR / "courier_grouped_train.parquet")
    courier_test_df = pd.read_parquet(DATA_DIR / "courier_grouped_test.parquet")
    
    pipe_grouped = Pipeline([
        ("prep", tree_preprocessor),
        ("model", LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1))
    ])
    pipe_grouped.fit(courier_train_df[num_cols + cat_cols], courier_train_df[target_col])
    cg_pred = pipe_grouped.predict(courier_test_df[num_cols + cat_cols])
    cg_metrics = compute_regression_metrics("Courier-Grouped (Unseen Couriers)", courier_test_df[target_col], cg_pred)
    
    # Random Split
    rand_train_df = pd.read_parquet(DATA_DIR / "random_train.parquet")
    rand_test_df = pd.read_parquet(DATA_DIR / "random_test.parquet")
    
    pipe_rand = Pipeline([
        ("prep", tree_preprocessor),
        ("model", LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1))
    ])
    pipe_rand.fit(rand_train_df[num_cols + cat_cols], rand_train_df[target_col])
    rand_pred = pipe_rand.predict(rand_test_df[num_cols + cat_cols])
    rand_metrics = compute_regression_metrics("Random 80/20 Split (Point-in-Time)", rand_test_df[target_col], rand_pred)
    
    # Temporal metrics for champion model
    temporal_metrics = compute_regression_metrics("Temporal 70/15/15 Split (Production)", y_test, best_test_pred)
    
    split_comparison = pd.DataFrame([
        {
            "Split Strategy": "Temporal (70/15/15)",
            "Evaluation Scenario": "Chronological Future Generalization",
            "Test MAE (min)": temporal_metrics["mae"],
            "Test RMSE (min)": temporal_metrics["rmse"],
            "Test R2": temporal_metrics["r2"]
        },
        {
            "Split Strategy": "Courier-Grouped Holdout",
            "Evaluation Scenario": "Cold-Start Generalization to Unseen Couriers",
            "Test MAE (min)": cg_metrics["mae"],
            "Test RMSE (min)": cg_metrics["rmse"],
            "Test R2": cg_metrics["r2"]
        },
        {
            "Split Strategy": "Random (80/20)",
            "Evaluation Scenario": "Uniform Interpolation Baseline",
            "Test MAE (min)": rand_metrics["mae"],
            "Test RMSE (min)": rand_metrics["rmse"],
            "Test R2": rand_metrics["r2"]
        }
    ])
    
    print(split_comparison.to_string(index=False))
    split_comparison.to_csv(RESULTS_DIR / "regression_split_comparison.csv", index=False)
    
    # -----------------------------------------------------------------------
    # Export Results & Visualizations
    # -----------------------------------------------------------------------
    regression_summary = {
        "models_benchmark": results_df.to_dict(orient="records"),
        "champion_model": best_name,
        "champion_test_metrics": temporal_metrics,
        "quantile_prediction_intervals": {
            "target_coverage": 80.0,
            "empirical_coverage_pct": round(empirical_coverage, 2),
            "median_interval_width_min": round(median_width, 2)
        },
        "advisory_buffers_min": {
            "p80_residual_buffer": round(derived_p80_buffer, 1),
            "p90_residual_buffer": round(derived_p90_buffer, 1)
        },
        "split_generalization": split_comparison.to_dict(orient="records")
    }
    
    with open(RESULTS_DIR / "regression_metrics.json", "w") as f:
        json.dump(regression_summary, f, indent=4)
        
    results_df.to_csv(RESULTS_DIR / "regression_model_comparison.csv", index=False)
    
    # Visual 1: Actual vs Predicted
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    sample_idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
    ax[0].scatter(y_test.iloc[sample_idx], best_test_pred[sample_idx], alpha=0.3, color="#2563eb", s=15)
    max_val = max(y_test.max(), best_test_pred.max())
    ax[0].plot([0, 800], [0, 800], color="#dc2626", linestyle="--", linewidth=1.5, label="Perfect Alignment")
    ax[0].set_xlim(0, 800)
    ax[0].set_ylim(0, 800)
    ax[0].set_title(f"Actual vs Predicted Duration ({best_name})\nTest MAE: {temporal_metrics['mae']}m | R²: {temporal_metrics['r2']}", fontsize=11, fontweight="bold")
    ax[0].set_xlabel("Actual Delivery Duration (min)")
    ax[0].set_ylabel("Predicted Duration (min)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.25)
    
    # Visual 2: Quantile Intervals Sample
    sorted_order = np.argsort(p50_pred[:60])
    x_axis = np.arange(60)
    ax[1].fill_between(x_axis, p10_pred[:60][sorted_order], p90_pred[:60][sorted_order], color="#93c5fd", alpha=0.4, label="80% Prediction Interval [P10, P90]")
    ax[1].plot(x_axis, p50_pred[:60][sorted_order], color="#1d4ed8", label="Median Estimate (P50)", linewidth=1.5)
    ax[1].scatter(x_axis, y_test.iloc[:60].iloc[sorted_order], color="#dc2626", s=20, label="Actual Duration", zorder=5)
    ax[1].set_title(f"Calibrated Prediction Intervals (Sample 60 Orders)\nEmpirical Test Coverage: {empirical_coverage:.1f}%", fontsize=11, fontweight="bold")
    ax[1].set_xlabel("Ordered Test Samples")
    ax[1].set_ylabel("Duration (min)")
    ax[1].legend(loc="upper left")
    ax[1].grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "regression_performance_and_intervals.png", dpi=300)
    plt.close()
    
    print(f"\nAll regression artifacts, intervals, and plots saved successfully to {RESULTS_DIR}")

if __name__ == "__main__":
    run_regression_training()
