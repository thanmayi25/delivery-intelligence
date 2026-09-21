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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE) Calculation
# ---------------------------------------------------------------------------
def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

# ---------------------------------------------------------------------------
# Metric Evaluator
# ---------------------------------------------------------------------------
def compute_classification_metrics(name, y_true, y_pred, y_prob, fit_time=0.0):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else float(np.mean(y_true))
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true.to_numpy() if hasattr(y_true, "to_numpy") else y_true, y_prob)
    
    return {
        "model_name": name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "fit_time_seconds": round(float(fit_time), 2)
    }

def run_classification_training():
    print("=" * 75)
    print("HIGH-DELAY RISK CLASSIFICATION BENCHMARK (TEMPORAL SPLIT)")
    print("=" * 75)
    
    train_df = pd.read_parquet(DATA_DIR / "train_features.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val_features.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_features.parquet")
    
    with open(DATA_DIR / "metadata.json", "r") as f:
        meta = json.load(f)
        
    num_cols = meta["numerical_cols"]
    cat_cols = meta["categorical_cols"]
    target_col = "high_delay"
    
    X_train = train_df[num_cols + cat_cols]
    y_train = train_df[target_col]
    X_val = val_df[num_cols + cat_cols]
    y_val = val_df[target_col]
    X_test = test_df[num_cols + cat_cols]
    y_test = test_df[target_col]
    
    pos_rate_train = float(y_train.mean())
    scale_weight = (1.0 - pos_rate_train) / pos_rate_train
    print(f"Train samples: {len(X_train)} (Positive rate: {pos_rate_train*100:.2f}%)")
    print(f"Val samples:   {len(X_val)} (Positive rate: {y_val.mean()*100:.2f}%)")
    print(f"Test samples:  {len(X_test)} (Positive rate: {y_test.mean()*100:.2f}%)")
    print(f"Computed scale_pos_weight: {scale_weight:.2f}")
    
    results = []
    
    # 1. Baseline 1: Majority Class (All 0)
    y_pred_maj = np.zeros(len(y_test), dtype=int)
    y_prob_maj = np.full(len(y_test), pos_rate_train)
    results.append(compute_classification_metrics("Majority Class Baseline (Always 0)", y_test, y_pred_maj, y_prob_maj))
    
    # 2. Baseline 2: Stratified Prior Baseline
    np.random.seed(42)
    y_pred_strat = np.random.binomial(1, pos_rate_train, size=len(y_test))
    results.append(compute_classification_metrics("Stratified Prior Baseline", y_test, y_pred_strat, y_prob_maj))
    
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
    
    classifiers = {
        "Logistic Regression (Balanced Baseline)": Pipeline([
            ("prep", linear_preprocessor),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "Random Forest Classifier": Pipeline([
            ("prep", tree_preprocessor),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", min_samples_leaf=10, random_state=42, n_jobs=-1))
        ]),
        "XGBoost Classifier": Pipeline([
            ("prep", tree_preprocessor),
            ("model", XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=scale_weight, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="logloss"))
        ]),
        "LightGBM Classifier": Pipeline([
            ("prep", tree_preprocessor),
            ("model", LGBMClassifier(n_estimators=100, max_depth=6, num_leaves=31, scale_pos_weight=scale_weight, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1))
        ])
    }
    
    fitted_models = {}
    test_probs = {}
    test_preds = {}
    
    for name, pipe in classifiers.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)
        
        metrics = compute_classification_metrics(name, y_test, y_pred, y_prob, fit_time)
        results.append(metrics)
        fitted_models[name] = pipe
        test_probs[name] = y_prob
        test_preds[name] = y_pred
        
        print(f" -> ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']} | F1: {metrics['f1_score']} | Brier: {metrics['brier_score']}")
        
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 75)
    print("CLASSIFICATION BENCHMARK COMPARISON (TEST SET):")
    print("=" * 75)
    print(results_df.to_string(index=False))
    
    best_name = "LightGBM Classifier"
    best_pipe = fitted_models[best_name]
    joblib.dump(best_pipe, MODELS_DIR / "best_delay_classifier.joblib")
    
    # -----------------------------------------------------------------------
    # Probability Calibration (Fit on Validation Split)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 2: PROBABILITY CALIBRATION (ISOTONIC & PLATT/SIGMOID ON VALIDATION SET)")
    print("=" * 75)
    
    # Sigmoid (Platt Scaling) Calibrator fit strictly on Validation set
    platt_calibrator = CalibratedClassifierCV(estimator=best_pipe, method="sigmoid", cv="prefit")
    platt_calibrator.fit(X_val, y_val)
    prob_test_platt = platt_calibrator.predict_proba(X_test)[:, 1]
    pred_test_platt = (prob_test_platt >= 0.5).astype(int)
    platt_metrics = compute_classification_metrics("LightGBM + Platt (Sigmoid) Calibrated", y_test, pred_test_platt, prob_test_platt)
    
    # Isotonic Calibrator fit strictly on Validation set
    isotonic_calibrator = CalibratedClassifierCV(estimator=best_pipe, method="isotonic", cv="prefit")
    isotonic_calibrator.fit(X_val, y_val)
    prob_test_iso = isotonic_calibrator.predict_proba(X_test)[:, 1]
    pred_test_iso = (prob_test_iso >= 0.5).astype(int)
    iso_metrics = compute_classification_metrics("LightGBM + Isotonic Calibrated", y_test, pred_test_iso, prob_test_iso)
    
    uncal_prob = test_probs[best_name]
    uncal_ece = compute_ece(y_test.to_numpy(), uncal_prob)
    platt_ece = compute_ece(y_test.to_numpy(), prob_test_platt)
    iso_ece = compute_ece(y_test.to_numpy(), prob_test_iso)
    
    uncal_brier = brier_score_loss(y_test, uncal_prob)
    platt_brier = brier_score_loss(y_test, prob_test_platt)
    iso_brier = brier_score_loss(y_test, prob_test_iso)
    
    print(f"Uncalibrated LightGBM:    Brier = {uncal_brier:.4f} | ECE = {uncal_ece:.4f}")
    print(f"Platt (Sigmoid) Calibrated: Brier = {platt_brier:.4f} | ECE = {platt_ece:.4f}")
    print(f"Isotonic Calibrated:        Brier = {iso_brier:.4f} | ECE = {iso_ece:.4f}")
    
    # Save the calibrated champion model (Platt scaling is typically smoother and more robust)
    champion_calibrator = platt_calibrator if platt_brier <= iso_brier else isotonic_calibrator
    champion_calibrator_name = "Platt (Sigmoid)" if platt_brier <= iso_brier else "Isotonic"
    joblib.dump(champion_calibrator, MODELS_DIR / "calibrated_delay_classifier.joblib")
    print(f"Saved best calibrated classifier ({champion_calibrator_name}) to {MODELS_DIR / 'calibrated_delay_classifier.joblib'}")
    
    # -----------------------------------------------------------------------
    # Decision Threshold Analysis (Validation Set Tuning)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 3: DECISION THRESHOLD ANALYSIS")
    print("=" * 75)
    
    val_prob = platt_calibrator.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores_val = [f1_score(y_val, (val_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_thresh_idx = np.argmax(f1_scores_val)
    optimal_thresh = float(thresholds[best_thresh_idx])
    max_val_f1 = float(f1_scores_val[best_thresh_idx])
    
    test_pred_opt = (prob_test_platt >= optimal_thresh).astype(int)
    opt_test_f1 = float(f1_score(y_test, test_pred_opt, zero_division=0))
    opt_test_prec = float(precision_score(y_test, test_pred_opt, zero_division=0))
    opt_test_rec = float(recall_score(y_test, test_pred_opt, zero_division=0))
    
    print(f"Validation F1-Optimal Threshold: {optimal_thresh:.2f} (Val F1: {max_val_f1:.4f})")
    print(f"Test Set Performance at Threshold {optimal_thresh:.2f}:")
    print(f" - Precision: {opt_test_prec:.4f} | Recall: {opt_test_rec:.4f} | F1: {opt_test_f1:.4f}")
    
    # -----------------------------------------------------------------------
    # Multi-Split Generalization Benchmark (Classification)
    # -----------------------------------------------------------------------
    courier_train_df = pd.read_parquet(DATA_DIR / "courier_grouped_train.parquet")
    courier_test_df = pd.read_parquet(DATA_DIR / "courier_grouped_test.parquet")
    
    pipe_grouped = Pipeline([
        ("prep", tree_preprocessor),
        ("model", LGBMClassifier(n_estimators=100, max_depth=6, scale_pos_weight=scale_weight, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1))
    ])
    pipe_grouped.fit(courier_train_df[num_cols + cat_cols], courier_train_df[target_col])
    cg_prob = pipe_grouped.predict_proba(courier_test_df[num_cols + cat_cols])[:, 1]
    cg_pred = pipe_grouped.predict(courier_test_df[num_cols + cat_cols])
    cg_metrics = compute_classification_metrics("Courier-Grouped (Unseen Couriers)", courier_test_df[target_col], cg_pred, cg_prob)
    
    rand_train_df = pd.read_parquet(DATA_DIR / "random_train.parquet")
    rand_test_df = pd.read_parquet(DATA_DIR / "random_test.parquet")
    
    pipe_rand = Pipeline([
        ("prep", tree_preprocessor),
        ("model", LGBMClassifier(n_estimators=100, max_depth=6, scale_pos_weight=scale_weight, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1))
    ])
    pipe_rand.fit(rand_train_df[num_cols + cat_cols], rand_train_df[target_col])
    rand_prob = pipe_rand.predict_proba(rand_test_df[num_cols + cat_cols])[:, 1]
    rand_pred = pipe_rand.predict(rand_test_df[num_cols + cat_cols])
    rand_metrics = compute_classification_metrics("Random 80/20 Split (Point-in-Time)", rand_test_df[target_col], rand_pred, rand_prob)
    
    temporal_metrics = compute_classification_metrics("Temporal 70/15/15 Split (Production)", y_test, test_preds[best_name], test_probs[best_name])
    
    split_cls_comp = pd.DataFrame([
        {
            "Split Strategy": "Temporal (70/15/15)",
            "Evaluation Scenario": "Chronological Future Generalization",
            "ROC-AUC": temporal_metrics["roc_auc"],
            "PR-AUC": temporal_metrics["pr_auc"],
            "F1-Score": temporal_metrics["f1_score"],
            "Brier Score": temporal_metrics["brier_score"]
        },
        {
            "Split Strategy": "Courier-Grouped Holdout",
            "Evaluation Scenario": "Cold-Start Generalization to Unseen Couriers",
            "ROC-AUC": cg_metrics["roc_auc"],
            "PR-AUC": cg_metrics["pr_auc"],
            "F1-Score": cg_metrics["f1_score"],
            "Brier Score": cg_metrics["brier_score"]
        },
        {
            "Split Strategy": "Random (80/20)",
            "Evaluation Scenario": "Uniform Interpolation Baseline",
            "ROC-AUC": rand_metrics["roc_auc"],
            "PR-AUC": rand_metrics["pr_auc"],
            "F1-Score": rand_metrics["f1_score"],
            "Brier Score": rand_metrics["brier_score"]
        }
    ])
    split_cls_comp.to_csv(RESULTS_DIR / "classification_split_comparison.csv", index=False)
    
    # -----------------------------------------------------------------------
    # Export Results & Visualizations
    # -----------------------------------------------------------------------
    cls_summary = {
        "models_benchmark": results_df.to_dict(orient="records"),
        "champion_model": best_name,
        "champion_test_metrics": temporal_metrics,
        "calibration": {
            "uncalibrated_brier": round(uncal_brier, 4),
            "uncalibrated_ece": round(uncal_ece, 4),
            "platt_brier": round(platt_brier, 4),
            "platt_ece": round(platt_ece, 4),
            "isotonic_brier": round(iso_brier, 4),
            "isotonic_ece": round(iso_ece, 4),
            "recommended_calibrator": champion_calibrator_name
        },
        "threshold_tuning": {
            "optimal_threshold": round(optimal_thresh, 2),
            "val_f1_at_optimal": round(max_val_f1, 4),
            "test_f1_at_optimal": round(opt_test_f1, 4),
            "test_precision_at_optimal": round(opt_test_prec, 4),
            "test_recall_at_optimal": round(opt_test_rec, 4)
        },
        "split_generalization": split_cls_comp.to_dict(orient="records")
    }
    
    with open(RESULTS_DIR / "classification_metrics.json", "w") as f:
        json.dump(cls_summary, f, indent=4)
        
    results_df.to_csv(RESULTS_DIR / "classification_model_comparison.csv", index=False)
    
    # Visualizations: ROC Curves & Calibration Curves
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: ROC Curves
    for name, prob in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.5)
        
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (0.50)")
    ax[0].set_title("ROC Curves (Temporal Test Split)", fontsize=11, fontweight="bold")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].grid(True, alpha=0.25)
    
    # Subplot 2: Calibration Curves
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, uncal_prob, n_bins=8)
    prob_true_platt, prob_pred_platt = calibration_curve(y_test, prob_test_platt, n_bins=8)
    prob_true_iso, prob_pred_iso = calibration_curve(y_test, prob_test_iso, n_bins=8)
    
    ax[1].plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax[1].plot(prob_pred_uncal, prob_true_uncal, "s-", color="#dc2626", label=f"Uncalibrated LightGBM (ECE={uncal_ece:.3f})")
    ax[1].plot(prob_pred_platt, prob_true_platt, "o-", color="#2563eb", label=f"Platt Calibrated (ECE={platt_ece:.3f})")
    ax[1].plot(prob_pred_iso, prob_true_iso, "^-", color="#16a34a", label=f"Isotonic Calibrated (ECE={iso_ece:.3f})")
    ax[1].set_title("Probability Calibration Curves (Reliability Diagram)", fontsize=11, fontweight="bold")
    ax[1].set_xlabel("Mean Predicted Probability")
    ax[1].set_ylabel("Fraction of Positives")
    ax[1].legend(loc="upper left", fontsize=8)
    ax[1].grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "classification_calibration_and_roc.png", dpi=300)
    plt.close()
    
    print(f"\nAll classification artifacts and calibration curves saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_classification_training()
