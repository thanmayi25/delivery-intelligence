from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SHAP_DIR = RESULTS_DIR / "shap"

SHAP_DIR.mkdir(parents=True, exist_ok=True)

def run_shap_analysis():
    print("=" * 75)
    print("PHASE 2: TREESHAP EXPLAINABILITY & FEATURE ATTRIBUTION")
    print("=" * 75)
    
    test_df = pd.read_parquet(DATA_DIR / "test_features.parquet")
    with open(DATA_DIR / "metadata.json", "r") as f:
        meta = json.load(f)
        
    num_cols = meta["numerical_cols"]
    cat_cols = meta["categorical_cols"]
    feature_cols = num_cols + cat_cols
    
    # Load champion pipeline
    reg_pipe = joblib.load(MODELS_DIR / "best_duration_regressor.joblib")
    preprocessor = reg_pipe.named_steps["prep"]
    lgbm_model = reg_pipe.named_steps["model"]
    
    # Transform test set features
    X_test_raw = test_df[feature_cols]
    X_test_transformed = preprocessor.transform(X_test_raw)
    
    # Extract feature names after OneHotEncoding
    ohe_cols = list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    all_feature_names = num_cols + ohe_cols
    
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)
    
    # Sample 1,000 background/eval points for TreeSHAP
    sample_n = min(1000, len(X_test_df))
    eval_df = X_test_df.iloc[:sample_n]
    
    print(f"Computing TreeSHAP values for {sample_n} test instances across {len(all_feature_names)} features...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer(eval_df)
    
    # 1. Global Feature Importance (Mean Absolute SHAP)
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "mean_abs_shap_impact_minutes": np.round(mean_abs_shap, 2)
    }).sort_values("mean_abs_shap_impact_minutes", ascending=False).reset_index(drop=True)
    
    print("\nTOP 10 FEATURES BY MEAN ABSOLUTE SHAP IMPACT (MINUTES):")
    print(shap_importance_df.head(10).to_string(index=False))
    shap_importance_df.to_csv(SHAP_DIR / "shap_feature_importance.csv", index=False)
    
    # 2. SHAP Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("TreeSHAP Global Feature Impact on Delivery Duration (Test Set)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. SHAP Summary Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("Mean Absolute SHAP Value (Impact on Delivery Duration in Minutes)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. SHAP Dependence Plot: Delivery Distance & Active In-Flight Tasks
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    if "delivery_distance_km" in all_feature_names:
        shap.plots.scatter(shap_values[:, "delivery_distance_km"], color=shap_values[:, "active_inflight_tasks"], ax=ax[0], show=False)
        ax[0].set_title("SHAP Dependence: Delivery Distance (km)", fontweight="bold")
    if "duration_of_most_recently_completed_task" in all_feature_names:
        shap.plots.scatter(shap_values[:, "duration_of_most_recently_completed_task"], ax=ax[1], show=False)
        ax[1].set_title("SHAP Dependence: Prior Completed Task Duration (min)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_dependence.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. Local Waterfall Explanations (High vs Low Duration Cases)
    high_dur_idx = int(np.argmax(test_df["delivery_duration_minutes"].iloc[:sample_n]))
    low_dur_idx = int(np.argmin(test_df["delivery_duration_minutes"].iloc[:sample_n]))
    
    # Waterfall for high delay case
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[high_dur_idx], max_display=10, show=False)
    plt.title(f"Local SHAP Explanation: High Duration Delivery (Actual: {test_df['delivery_duration_minutes'].iloc[high_dur_idx]:.0f}m)", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_waterfall_high_delay.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Waterfall for optimal case
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[low_dur_idx], max_display=10, show=False)
    plt.title(f"Local SHAP Explanation: Optimal Rapid Delivery (Actual: {test_df['delivery_duration_minutes'].iloc[low_dur_idx]:.0f}m)", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_waterfall_low_delay.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Export summary metadata
    shap_summary = {
        "evaluation_samples": sample_n,
        "base_value_minutes": round(float(shap_values.base_values[0]), 2),
        "top_features": shap_importance_df.head(10).to_dict(orient="records")
    }
    with open(SHAP_DIR / "shap_summary.json", "w") as f:
        json.dump(shap_summary, f, indent=4)
        
    print(f"\nAll SHAP plots and importance summaries successfully saved to {SHAP_DIR}")

if __name__ == "__main__":
    run_shap_analysis()
