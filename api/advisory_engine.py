from pathlib import Path
import json
from typing import Dict, Any, List

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
    def default_rules() -> Dict[str, Any]:
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
        
    def evaluate_task_sequence(
        self,
        prev_completed_duration: float,
        jump_dist_km: float,
        active_inflight: int,
        current_distance_km: float,
        workload_1h: int = 0
    ) -> Dict[str, Any]:
        risk_score = 0
        advisories: List[str] = []
        flags: List[str] = []
        
        # 1. Evaluate Prior Completed Task Association
        if prev_completed_duration >= self.rules.get("completed_duration_thresholds_min", {}).get("critical", 380.0):
            risk_score += 40
            flags.append("CRITICAL_PRIOR_COMPLETION_DELAY")
            p80_buf = self.rules.get("recommended_buffers_min", {}).get("p80_delay_buffer", 175.0)
            advisories.append(
                f"⚠️ Preceding completed task duration exceeded empirical P90 (≥380m). "
                f"Statistical association indicates elevated probability of shift overrun. "
                f"Data-driven advisory buffer: +{p80_buf:.0f}m."
            )
        elif prev_completed_duration >= self.rules.get("completed_duration_thresholds_min", {}).get("elevated", 275.0):
            risk_score += 25
            flags.append("ELEVATED_PRIOR_COMPLETION_DELAY")
            advisories.append(
                "⚠️ Preceding completed task was above P75 (≥275m). Courier shift shows moderate historical latency."
            )
        elif prev_completed_duration >= self.rules.get("completed_duration_thresholds_min", {}).get("moderate", 175.0):
            risk_score += 10
            flags.append("MODERATE_PRIOR_COMPLETION_DELAY")
            advisories.append("ℹ️ Preceding completed delivery duration was near median (~175m). Standard operating conditions.")
            
        # 2. Evaluate Active In-Flight Workload (Wave Congestion)
        if active_inflight >= self.rules.get("active_inflight_thresholds", {}).get("heavy_wave", 15):
            risk_score += 25
            flags.append("HEAVY_INFLIGHT_WAVE")
            advisories.append(f"📦 High concurrent load: {active_inflight} active parcels in flight for this courier.")
        elif active_inflight >= self.rules.get("active_inflight_thresholds", {}).get("moderate_wave", 8):
            risk_score += 10
            flags.append("MODERATE_INFLIGHT_WAVE")
            
        # 3. Evaluate Spatial Transition Distance
        if jump_dist_km >= self.rules.get("spatial_jump_thresholds_km", {}).get("high_friction", 2.5):
            risk_score += 20
            flags.append("HIGH_TRANSITION_DISTANCE")
            advisories.append(f"📍 Substantial repositioning distance ({jump_dist_km:.1f} km from last pickup location).")
        elif jump_dist_km >= self.rules.get("spatial_jump_thresholds_km", {}).get("moderate_friction", 1.0):
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
