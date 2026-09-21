from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

def run_subgroup_analysis():
    print("=" * 75)
    print("PHASE 2: SUBGROUP & ERROR SLICING ANALYSIS (TEST SET)")
    print("=" * 75)
    
    test_df = pd.read_parquet(DATA_DIR / "test_features.parquet")
    with open(DATA_DIR / "metadata.json", "r") as f:
        meta = json.load(f)
        
    num_cols = meta["numerical_cols"]
    cat_cols = meta["categorical_cols"]
    feature_cols = num_cols + cat_cols
    
    reg_pipe = joblib.load(MODELS_DIR / "best_duration_regressor.joblib")
    cls_pipe = joblib.load(MODELS_DIR / "calibrated_delay_classifier.joblib")
    
    test_df["predicted_duration"] = reg_pipe.predict(test_df[feature_cols])
    test_df["delay_prob"] = cls_pipe.predict_proba(test_df[feature_cols])[:, 1]
    test_df["abs_error"] = (test_df["delivery_duration_minutes"] - test_df["predicted_duration"]).abs()
    test_df["signed_error"] = test_df["delivery_duration_minutes"] - test_df["predicted_duration"]
    
    subgroups = {}
    
    # 1. Subgroup: Distance Bins
    dist_bins = [0, 1.0, 2.5, 5.0, np.inf]
    dist_labels = ["< 1.0 km", "1.0 - 2.5 km", "2.5 - 5.0 km", "> 5.0 km"]
    test_df["distance_tier"] = pd.cut(test_df["delivery_distance_km"], bins=dist_bins, labels=dist_labels)
    
    dist_summary = test_df.groupby("distance_tier", observed=False).agg(
        sample_count=("order_id", "count"),
        actual_median_dur=("delivery_duration_minutes", "median"),
        pred_median_dur=("predicted_duration", "median"),
        mae=("abs_error", "mean"),
        mean_signed_bias=("signed_error", "mean"),
        high_delay_rate=("high_delay", "mean")
    ).reset_index()
    subgroups["distance_tiers"] = dist_summary.to_dict(orient="records")
    
    # 2. Subgroup: Time Period
    time_summary = test_df.groupby("time_period").agg(
        sample_count=("order_id", "count"),
        actual_median_dur=("delivery_duration_minutes", "median"),
        pred_median_dur=("predicted_duration", "median"),
        mae=("abs_error", "mean"),
        mean_signed_bias=("signed_error", "mean"),
        high_delay_rate=("high_delay", "mean")
    ).reset_index()
    subgroups["time_periods"] = time_summary.to_dict(orient="records")
    
    # 3. Subgroup: Active In-Flight Workload
    inflight_bins = [-1, 3, 7, 15, np.inf]
    inflight_labels = ["Low (0-3)", "Medium (4-7)", "High (8-15)", "Heavy Wave (16+)"]
    test_df["inflight_tier"] = pd.cut(test_df["active_inflight_tasks"], bins=inflight_bins, labels=inflight_labels)
    
    inflight_summary = test_df.groupby("inflight_tier", observed=False).agg(
        sample_count=("order_id", "count"),
        actual_median_dur=("delivery_duration_minutes", "median"),
        pred_median_dur=("predicted_duration", "median"),
        mae=("abs_error", "mean"),
        mean_signed_bias=("signed_error", "mean"),
        high_delay_rate=("high_delay", "mean")
    ).reset_index()
    subgroups["inflight_workload_tiers"] = inflight_summary.to_dict(orient="records")
    
    # 4. Subgroup: Delivery Regions
    region_summary = test_df.groupby("region_id").agg(
        sample_count=("order_id", "count"),
        actual_median_dur=("delivery_duration_minutes", "median"),
        pred_median_dur=("predicted_duration", "median"),
        mae=("abs_error", "mean"),
        mean_signed_bias=("signed_error", "mean"),
        high_delay_rate=("high_delay", "mean")
    ).reset_index()
    subgroups["regions"] = region_summary.to_dict(orient="records")
    
    print("\n1. PERFORMANCE BY DISTANCE TIER:")
    print(dist_summary.to_string(index=False))
    
    print("\n2. PERFORMANCE BY TIME OF DAY:")
    print(time_summary.to_string(index=False))
    
    print("\n3. PERFORMANCE BY ACTIVE IN-FLIGHT WORKLOAD:")
    print(inflight_summary.to_string(index=False))
    
    print("\n4. PERFORMANCE BY REGION:")
    print(region_summary.to_string(index=False))
    
    with open(RESULTS_DIR / "subgroup_error_analysis.json", "w") as f:
        json.dump(subgroups, f, indent=4)
        
    # Plot Subgroup Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distances
    axes[0, 0].bar(dist_summary["distance_tier"].astype(str), dist_summary["mae"], color="#2563eb", alpha=0.8, edgecolor="black")
    axes[0, 0].set_title("MAE by Delivery Distance Tier", fontweight="bold")
    axes[0, 0].set_ylabel("Mean Absolute Error (min)")
    axes[0, 0].grid(True, alpha=0.25)
    
    # Time Period
    axes[0, 1].bar(time_summary["time_period"].astype(str), time_summary["mae"], color="#16a34a", alpha=0.8, edgecolor="black")
    axes[0, 1].set_title("MAE by Time of Day", fontweight="bold")
    axes[0, 1].set_ylabel("Mean Absolute Error (min)")
    axes[0, 1].grid(True, alpha=0.25)
    
    # In-Flight Workload
    axes[1, 0].bar(inflight_summary["inflight_tier"].astype(str), inflight_summary["mae"], color="#f59e0b", alpha=0.8, edgecolor="black")
    axes[1, 0].set_title("MAE by Active In-Flight Workload", fontweight="bold")
    axes[1, 0].set_ylabel("Mean Absolute Error (min)")
    axes[1, 0].grid(True, alpha=0.25)
    
    # Regions
    axes[1, 1].bar([f"Region {r}" for r in region_summary["region_id"]], region_summary["mae"], color="#8b5cf6", alpha=0.8, edgecolor="black")
    axes[1, 1].set_title("MAE by Geographic Region", fontweight="bold")
    axes[1, 1].set_ylabel("Mean Absolute Error (min)")
    axes[1, 1].grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "subgroup_error_analysis.png", dpi=300)
    plt.close()
    
    print(f"\nSubgroup error analysis saved to {RESULTS_DIR / 'subgroup_error_analysis.json'}")

if __name__ == "__main__":
    run_subgroup_analysis()
