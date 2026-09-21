from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib

from api.schemas import (
    OrderFeatureInput,
    RawGPSOrderInput,
    PredictionInterval,
    DispatchEvaluationResponse,
    SequenceAdvisoryOutput,
    BatchDispatchEvaluationResponse
)
from api.advisory_engine import SequenceAdvisoryEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

def haversine_distance_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c

def assign_time_period(hour: int) -> str:
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

class DeliveryIntelligenceService:
    def __init__(self):
        self.reg_model_path = MODELS_DIR / "best_duration_regressor.joblib"
        self.quantile_models_path = MODELS_DIR / "quantile_regressors.joblib"
        self.clf_model_path = MODELS_DIR / "calibrated_delay_classifier.joblib"
        self.rules_path = MODELS_DIR / "sequence_advisory_rules.json"
        
        self.reg_model = None
        self.quantile_models = {}
        self.clf_model = None
        self.advisory_engine = None
        self.rules = {}
        self.optimal_decision_threshold = 0.20 # Derived from validation F1 tuning
        
        self.load_artifacts()
        
    def load_artifacts(self):
        if self.reg_model_path.exists():
            self.reg_model = joblib.load(self.reg_model_path)
            
        if self.quantile_models_path.exists():
            self.quantile_models = joblib.load(self.quantile_models_path)
            
        if self.clf_model_path.exists():
            self.clf_model = joblib.load(self.clf_model_path)
            
        if self.rules_path.exists():
            with open(self.rules_path, "r") as f:
                self.rules = json.load(f)
                
        self.advisory_engine = SequenceAdvisoryEngine(rules_path=self.rules_path)

    def evaluate_single_order(self, order: OrderFeatureInput) -> DispatchEvaluationResponse:
        input_dict = order.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # 1. Point ETA Prediction
        pred_duration = float(self.reg_model.predict(input_df)[0])
        pred_duration = max(1.0, pred_duration)
        
        # 2. Quantile Prediction Intervals (P10, P50, P90)
        if self.quantile_models and "p10" in self.quantile_models and "p90" in self.quantile_models:
            p10 = float(max(0.0, self.quantile_models["p10"].predict(input_df)[0]))
            p50 = float(max(0.0, self.quantile_models["p50"].predict(input_df)[0]))
            p90 = float(max(p10, self.quantile_models["p90"].predict(input_df)[0]))
        else:
            p10 = max(0.0, pred_duration * 0.5)
            p50 = pred_duration
            p90 = pred_duration * 1.8
            
        pred_interval = PredictionInterval(
            p10_minutes=round(p10, 1),
            p50_median_minutes=round(p50, 1),
            p90_minutes=round(p90, 1),
            interval_coverage_pct=80.0
        )
        
        # 3. Calibrated Delay Risk
        pred_proba = float(self.clf_model.predict_proba(input_df)[0][1])
        is_delayed = bool(pred_proba >= self.optimal_decision_threshold)
        
        risk_level = "CRITICAL" if pred_proba >= 0.35 else ("ELEVATED" if pred_proba >= self.optimal_decision_threshold else "LOW")
        
        # 4. Sequence Advisory Engine
        adv = self.advisory_engine.evaluate_task_sequence(
            prev_completed_duration=order.duration_of_most_recently_completed_task,
            jump_dist_km=order.previous_accept_distance_km,
            active_inflight=order.active_inflight_tasks,
            current_distance_km=order.delivery_distance_km,
            workload_1h=order.tasks_previous_1h
        )
        
        seq_output = SequenceAdvisoryOutput(
            risk_score=adv["risk_score"],
            risk_tier=adv["risk_tier"],
            badge_color=adv["badge_color"],
            flags=adv["flags"],
            advisories=adv["advisories"]
        )
        
        return DispatchEvaluationResponse(
            predicted_duration_minutes=round(pred_duration, 1),
            predicted_duration_hours=round(pred_duration / 60.0, 2),
            prediction_interval=pred_interval,
            high_delay_risk_probability=round(pred_proba, 4),
            is_high_delay_flag=is_delayed,
            risk_level=risk_level,
            sequence_advisory=seq_output,
            metadata={
                "model_version": "v2.0-PointInTime-TemporalSplit",
                "calibration_type": "Isotonic/Platt Calibrated",
                "high_delay_threshold_minutes": 380.0,
                "optimal_decision_threshold": self.optimal_decision_threshold
            }
        )

    def evaluate_raw_gps_order(self, raw_order: RawGPSOrderInput) -> DispatchEvaluationResponse:
        dist_km = haversine_distance_km(
            raw_order.accept_gps_lng, raw_order.accept_gps_lat,
            raw_order.delivery_gps_lng, raw_order.delivery_gps_lat
        )
        t = pd.to_datetime(raw_order.accept_time_str)
        hour = t.hour
        minute = t.minute
        weekday = t.dayofweek
        is_wknd = 1 if weekday >= 5 else 0
        t_period = assign_time_period(hour)
        
        feature_input = OrderFeatureInput(
            delivery_distance_km=float(dist_km),
            tasks_previous_1h=raw_order.tasks_previous_1h or 2,
            tasks_previous_3h=raw_order.tasks_previous_3h or 6,
            active_inflight_tasks=raw_order.active_inflight_tasks or 5,
            daily_task_index=3,
            is_first_task_of_day=0,
            minutes_since_last_dispatch_today=0.0,
            duration_of_most_recently_completed_task=raw_order.duration_of_most_recently_completed_task or 175.0,
            has_prior_completed_task=1,
            mins_since_recent_completed=30.0,
            previous_accept_distance_km=raw_order.previous_accept_distance_km or 0.5,
            accept_hour=hour,
            accept_minute=minute,
            accept_weekday=weekday,
            is_weekend=is_wknd,
            time_period=t_period
        )
        return self.evaluate_single_order(feature_input)

    def evaluate_batch(self, orders: list[OrderFeatureInput]) -> BatchDispatchEvaluationResponse:
        evaluated = [self.evaluate_single_order(o) for o in orders]
        total = len(evaluated)
        avg_dur = float(np.mean([e.predicted_duration_minutes for e in evaluated])) if total > 0 else 0.0
        delay_cnt = sum(1 for e in evaluated if e.is_high_delay_flag)
        delay_rate = (delay_cnt / total * 100.0) if total > 0 else 0.0
        
        return BatchDispatchEvaluationResponse(
            total_orders=total,
            average_predicted_duration_minutes=round(avg_dur, 1),
            high_delay_count=delay_cnt,
            high_delay_rate_percent=round(delay_rate, 2),
            results=evaluated
        )

service_instance = DeliveryIntelligenceService()
