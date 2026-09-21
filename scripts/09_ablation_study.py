from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

def run_ablation_study():
    print("=" * 75)
    print("PHASE 2: FEATURE ABLATION STUDY (TEMPORAL TEST SPLIT)")
    print("=" * 75)
    
    train_df = pd.read_parquet(DATA_DIR / "train_features.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_features.parquet")
    
    with open(DATA_DIR / "metadata.json", "r") as f:
        meta = json.load(f)
        
    target_reg = "delivery_duration_minutes"
    target_cls = "high_delay"
    scale_weight = (1.0 - train_df[target_cls].mean()) / train_df[target_cls].mean()
    
    # Feature Subsets
    distance_features = ["delivery_distance_km"]
    temporal_features = ["accept_hour", "accept_minute", "accept_weekday", "is_weekend", "time_period"]
    workload_features = ["active_inflight_tasks", "tasks_previous_1h", "tasks_previous_3h", "daily_task_index", "is_first_task_of_day", "minutes_since_last_dispatch_today"]
    sequence_features = ["duration_of_most_recently_completed_task", "has_prior_completed_task", "mins_since_recent_completed", "previous_accept_distance_km"]
    
    configurations = {
        "1. Full Feature Matrix (All Features)": distance_features + temporal_features + workload_features + sequence_features,
        "2. Ablated: No Sequence Lookups": distance_features + temporal_features + workload_features,
        "3. Ablated: No Workload & Concurrency": distance_features + temporal_features + sequence_features,
        "4. Ablated: Distance + Temporal Only": distance_features + temporal_features,
        "5. Minimal Baseline: Distance Only": distance_features
    }
    
    ablation_results = []
    
    for config_name, feat_list in configurations.items():
        print(f"\nEvaluating Configuration: {config_name} ({len(feat_list)} features)")
        
        num_feats = [f for f in feat_list if f != "time_period"]
        cat_feats = [f for f in feat_list if f == "time_period"]
        
        transformers = [("num", "passthrough", num_feats)]
        if cat_feats:
            transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_feats))
            
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Regression
        reg_model = Pipeline([
            ("prep", preprocessor),
            ("model", LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1))
        ])
        reg_model.fit(train_df[feat_list], train_df[target_reg])
        y_reg_pred = reg_model.predict(test_df[feat_list])
        
        mae = mean_absolute_error(test_df[target_reg], y_reg_pred)
        rmse = np.sqrt(mean_squared_error(test_df[target_reg], y_reg_pred))
        r2 = r2_score(test_df[target_reg], y_reg_pred)
        
        # Classification
        cls_model = Pipeline([
            ("prep", preprocessor),
            ("model", LGBMClassifier(n_estimators=100, max_depth=6, scale_pos_weight=scale_weight, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1))
        ])
        cls_model.fit(train_df[feat_list], train_df[target_cls])
        y_cls_prob = cls_model.predict_proba(test_df[feat_list])[:, 1]
        
        roc_auc = roc_auc_score(test_df[target_cls], y_cls_prob)
        pr_auc = average_precision_score(test_df[target_cls], y_cls_prob)
        
        ablation_results.append({
            "Configuration": config_name,
            "Num Features": len(feat_list),
            "Regression MAE (min)": round(float(mae), 2),
            "Regression RMSE (min)": round(float(rmse), 2),
            "Regression R2": round(float(r2), 4),
            "Classification ROC-AUC": round(float(roc_auc), 4),
            "Classification PR-AUC": round(float(pr_auc), 4)
        })
        
    ablation_df = pd.DataFrame(ablation_results)
    print("\n" + "=" * 75)
    print("FEATURE ABLATION BENCHMARK SUMMARY:")
    print("=" * 75)
    print(ablation_df.to_string(index=False))
    
    ablation_df.to_csv(RESULTS_DIR / "ablation_study.csv", index=False)
    with open(RESULTS_DIR / "ablation_study.json", "w") as f:
        json.dump(ablation_results, f, indent=4)
        
    print(f"\nAblation study artifacts saved to {RESULTS_DIR / 'ablation_study.csv'}")

if __name__ == "__main__":
    run_ablation_study()
