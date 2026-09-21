from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class SequenceAdvisoryEngine:
    """
    Task-Sequence Intelligence & Dispatch Advisory Engine.
    Evaluates preceding completed task dynamics, spatial gaps, and workload contexts
    to generate data-driven risk ratings, empirical buffers, and dispatch advisories.
    Uses strict observational and probabilistic terminology (zero unjustified causal claims).
    """
    def __init__(self, rules_path=None):
        if rules_path and Path(rules_path).exists():
            with open(rules_path, "r") as f:
                self.rules = json.load(f)
        else:
            self.rules = self.default_rules()
            
    @staticmethod
    def default_rules():
        return {
            "metadata": {
                "engine_version": "2.0-PointInTime",
                "domain": "Last-Mile Express Logistics",
                "calibration_source": "Empirical Test Split Residuals & Quantiles"
            },
            "completed_duration_thresholds_min": {
                "critical": 380.0, # Empirical P90
                "elevated": 275.0, # Empirical P75
                "moderate": 175.0  # Empirical P50
            },
            "spatial_jump_thresholds_km": {
                "high_friction": 2.5,
                "moderate_friction": 1.0
            },
            "active_inflight_thresholds": {
                "heavy_wave": 15,
                "moderate_wave": 8
            },
            "recommended_buffers_min": {
                "p80_delay_buffer": 175.0,
                "p90_delay_buffer": 259.0
            }
        }
        
    def evaluate_task_sequence(self, prev_completed_duration, jump_dist_km, active_inflight, current_distance_km, workload_1h=0):
        """
        Evaluates a single task dispatch context and returns risk rating and non-causal advisories.
        """
        risk_score = 0
        advisories = []
        flags = []
        
        # 1. Evaluate Prior Completed Task Association
        if prev_completed_duration >= self.rules["completed_duration_thresholds_min"]["critical"]:
            risk_score += 40
            flags.append("CRITICAL_PRIOR_COMPLETION_DELAY")
            advisories.append(
                f"⚠️ Preceding completed task duration exceeded empirical P90 (≥380m). "
                f"Statistical association indicates elevated probability of shift overrun. "
                f"Data-driven advisory buffer: +{self.rules['recommended_buffers_min']['p80_delay_buffer']:.0f}m."
            )
        elif prev_completed_duration >= self.rules["completed_duration_thresholds_min"]["elevated"]:
            risk_score += 25
            flags.append("ELEVATED_PRIOR_COMPLETION_DELAY")
            advisories.append(
                "⚠️ Preceding completed task was above P75 (≥275m). Courier shift shows moderate historical latency."
            )
        elif prev_completed_duration >= self.rules["completed_duration_thresholds_min"]["moderate"]:
            risk_score += 10
            flags.append("MODERATE_PRIOR_COMPLETION_DELAY")
            advisories.append("ℹ️ Preceding completed delivery duration was near median (~175m). Standard operating conditions.")
            
        # 2. Evaluate Active In-Flight Workload (Wave Congestion)
        if active_inflight >= self.rules["active_inflight_thresholds"]["heavy_wave"]:
            risk_score += 25
            flags.append("HEAVY_INFLIGHT_WAVE")
            advisories.append(f"📦 High concurrent load: {active_inflight} active parcels in flight for this courier.")
        elif active_inflight >= self.rules["active_inflight_thresholds"]["moderate_wave"]:
            risk_score += 10
            flags.append("MODERATE_INFLIGHT_WAVE")
            
        # 3. Evaluate Spatial Transition Distance
        if jump_dist_km >= self.rules["spatial_jump_thresholds_km"]["high_friction"]:
            risk_score += 20
            flags.append("HIGH_TRANSITION_DISTANCE")
            advisories.append(f"📍 Substantial repositioning distance ({jump_dist_km:.1f} km from last pickup location).")
        elif jump_dist_km >= self.rules["spatial_jump_thresholds_km"]["moderate_friction"]:
            risk_score += 10
            flags.append("MODERATE_TRANSITION_DISTANCE")
            
        # 4. Long Distance Delivery Combined Context
        if current_distance_km > 5.0 and risk_score > 30:
            risk_score += 15
            flags.append("COMPOUNDED_DISTANCE_RISK")
            advisories.append("🚨 Extended delivery route (>5km) combined with elevated historical completion latency.")
            
        # Overall Risk Tier
        if risk_score >= 50:
            tier = "CRITICAL RISK"
            color = "#ef4444"
        elif risk_score >= 30:
            tier = "HIGH RISK"
            color = "#f97316"
        elif risk_score >= 15:
            tier = "MODERATE RISK"
            color = "#eab308"
        else:
            tier = "LOW RISK / OPTIMAL"
            color = "#22c55e"
            if not advisories:
                advisories.append("✅ Sequence conditions are optimal. No elevated delay associations detected.")
                
        return {
            "risk_score": min(100, risk_score),
            "risk_tier": tier,
            "badge_color": color,
            "flags": flags,
            "advisories": advisories
        }

def run_bootstrap_analysis():
    print("=" * 75)
    print("STATISTICAL BOOTSTRAP ANALYSIS (1,000 ITERATIONS, 95% CONFIDENCE INTERVALS)")
    print("=" * 75)
    
    test_df = pd.read_parquet(DATA_DIR / "test_features.parquet")
    with open(DATA_DIR / "metadata.json", "r") as f:
        meta = json.load(f)
        
    num_cols = meta["numerical_cols"]
    cat_cols = meta["categorical_cols"]
    X_test = test_df[num_cols + cat_cols]
    y_test_dur = test_df["delivery_duration_minutes"].to_numpy()
    y_test_delay = test_df["high_delay"].to_numpy()
    
    # Load champion models
    import joblib
    reg_pipe = joblib.load(MODELS_DIR / "best_duration_regressor.joblib")
    cls_pipe = joblib.load(MODELS_DIR / "calibrated_delay_classifier.joblib")
    
    pred_dur = reg_pipe.predict(X_test)
    prob_delay = cls_pipe.predict_proba(X_test)[:, 1]
    
    n_samples = len(test_df)
    n_bootstrap = 1000
    np.random.seed(42)
    
    mae_list = []
    rmse_list = []
    r2_list = []
    roc_auc_list = []
    pr_auc_list = []
    brier_list = []
    
    corr_pearson_list = []
    corr_spearman_list = []
    odds_ratio_list = []
    
    prior_dur = test_df["duration_of_most_recently_completed_task"].to_numpy()
    prior_is_high_delay = (prior_dur >= meta["high_delay_threshold_minutes"]).astype(int)
    
    print(f"Running {n_bootstrap} bootstrap resamples over {n_samples} test records...")
    for i in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Regression metrics
        y_d = y_test_dur[idx]
        p_d = pred_dur[idx]
        mae_list.append(mean_absolute_error(y_d, p_d))
        rmse_list.append(np.sqrt(mean_squared_error(y_d, p_d)))
        r2_list.append(r2_score(y_d, p_d))
        
        # Classification metrics
        y_b = y_test_delay[idx]
        p_b = prob_delay[idx]
        if len(np.unique(y_b)) > 1:
            roc_auc_list.append(roc_auc_score(y_b, p_b))
            pr_auc_list.append(average_precision_score(y_b, p_b))
        brier_list.append(brier_score_loss(y_b, p_b))
        
        # Sequence correlation (prior completed vs current duration)
        pr_d = prior_dur[idx]
        r_p, _ = stats.pearsonr(pr_d, y_d)
        r_s, _ = stats.spearmanr(pr_d, y_d)
        corr_pearson_list.append(r_p)
        corr_spearman_list.append(r_s)
        
        # Sequence Delay Odds Ratio
        pr_h = prior_is_high_delay[idx]
        # Contingency table: [a, b], [c, d]
        a = np.sum((pr_h == 1) & (y_b == 1)) + 0.5
        b = np.sum((pr_h == 1) & (y_b == 0)) + 0.5
        c = np.sum((pr_h == 0) & (y_b == 1)) + 0.5
        d = np.sum((pr_h == 0) & (y_b == 0)) + 0.5
        odds_ratio = (a * d) / (b * c)
        odds_ratio_list.append(odds_ratio)
        
    def ci95(arr):
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]
        
    bootstrap_results = {
        "bootstrap_iterations": n_bootstrap,
        "sample_size": n_samples,
        "regression": {
            "mae": {"point_estimate": round(float(np.mean(mae_list)), 2), "ci_95": ci95(mae_list)},
            "rmse": {"point_estimate": round(float(np.mean(rmse_list)), 2), "ci_95": ci95(rmse_list)},
            "r2": {"point_estimate": round(float(np.mean(r2_list)), 4), "ci_95": ci95(r2_list)}
        },
        "classification": {
            "roc_auc": {"point_estimate": round(float(np.mean(roc_auc_list)), 4), "ci_95": ci95(roc_auc_list)},
            "pr_auc": {"point_estimate": round(float(np.mean(pr_auc_list)), 4), "ci_95": ci95(pr_auc_list)},
            "brier_score": {"point_estimate": round(float(np.mean(brier_list)), 4), "ci_95": ci95(brier_list)}
        },
        "sequence_association": {
            "pearson_r": {"point_estimate": round(float(np.mean(corr_pearson_list)), 4), "ci_95": ci95(corr_pearson_list)},
            "spearman_rho": {"point_estimate": round(float(np.mean(corr_spearman_list)), 4), "ci_95": ci95(corr_spearman_list)},
            "delay_odds_ratio": {"point_estimate": round(float(np.mean(odds_ratio_list)), 2), "ci_95": ci95(odds_ratio_list)}
        }
    }
    
    print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS (TEST SET):")
    print(f" - Regression MAE:      {bootstrap_results['regression']['mae']['point_estimate']} min (95% CI: {bootstrap_results['regression']['mae']['ci_95']})")
    print(f" - Regression R²:       {bootstrap_results['regression']['r2']['point_estimate']} (95% CI: {bootstrap_results['regression']['r2']['ci_95']})")
    print(f" - Classification ROC-AUC: {bootstrap_results['classification']['roc_auc']['point_estimate']} (95% CI: {bootstrap_results['classification']['roc_auc']['ci_95']})")
    print(f" - Classification PR-AUC:  {bootstrap_results['classification']['pr_auc']['point_estimate']} (95% CI: {bootstrap_results['classification']['pr_auc']['ci_95']})")
    print(f" - Non-leaked Sequence Pearson r:   {bootstrap_results['sequence_association']['pearson_r']['point_estimate']} (95% CI: {bootstrap_results['sequence_association']['pearson_r']['ci_95']})")
    print(f" - Non-leaked Sequence Spearman rho: {bootstrap_results['sequence_association']['spearman_rho']['point_estimate']} (95% CI: {bootstrap_results['sequence_association']['spearman_rho']['ci_95']})")
    print(f" - High-Delay Odds Ratio: {bootstrap_results['sequence_association']['delay_odds_ratio']['point_estimate']}x (95% CI: {bootstrap_results['sequence_association']['delay_odds_ratio']['ci_95']})")
    
    with open(RESULTS_DIR / "bootstrap_confidence_intervals.json", "w") as f:
        json.dump(bootstrap_results, f, indent=4)
        
    # Save advisory rules
    engine = SequenceAdvisoryEngine()
    with open(MODELS_DIR / "sequence_advisory_rules.json", "w") as f:
        json.dump(engine.rules, f, indent=4)
    with open(RESULTS_DIR / "sequence_advisory_rules.json", "w") as f:
        json.dump(engine.rules, f, indent=4)
        
    # Visual: Bootstrap Distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].hist(mae_list, bins=25, color="#2563eb", edgecolor="black", alpha=0.7)
    axes[0].axvline(np.mean(mae_list), color="#dc2626", linestyle="--", linewidth=2, label=f"Mean: {np.mean(mae_list):.1f}m")
    axes[0].set_title("Bootstrap Test MAE Distribution\n(1,000 Resamples)", fontweight="bold")
    axes[0].set_xlabel("MAE (minutes)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    
    axes[1].hist(roc_auc_list, bins=25, color="#16a34a", edgecolor="black", alpha=0.7)
    axes[1].axvline(np.mean(roc_auc_list), color="#dc2626", linestyle="--", linewidth=2, label=f"Mean: {np.mean(roc_auc_list):.3f}")
    axes[1].set_title("Bootstrap Test ROC-AUC Distribution\n(1,000 Resamples)", fontweight="bold")
    axes[1].set_xlabel("ROC-AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    
    axes[2].hist(corr_pearson_list, bins=25, color="#f59e0b", edgecolor="black", alpha=0.7)
    axes[2].axvline(np.mean(corr_pearson_list), color="#dc2626", linestyle="--", linewidth=2, label=f"Mean r: {np.mean(corr_pearson_list):.3f}")
    axes[2].set_title("True Point-in-Time Sequence Correlation\n(Non-Leaked Pearson r)", fontweight="bold")
    axes[2].set_xlabel("Pearson Correlation (r)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bootstrap_distributions.png", dpi=300)
    plt.close()
    
    print(f"\nSaved bootstrap distributions and advisory rules to {RESULTS_DIR}")

if __name__ == "__main__":
    run_bootstrap_analysis()
