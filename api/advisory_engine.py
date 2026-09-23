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
                f"⚠️ Previous Delivery Delayed (>6.3 hrs): Driver was delayed on their last delivery, "
                f"which significantly increases the chance of another delay on this run. Suggested buffer: +{p80_buf:.0f} min."
            )
        elif prev_completed_duration >= self.rules.get("completed_duration_thresholds_min", {}).get("elevated", 275.0):
            risk_score += 25
            flags.append("ELEVATED_PRIOR_COMPLETION_DELAY")
            advisories.append(
                "⚠️ Longer Previous Delivery (~4.5 hrs): Driver's last run took longer than usual. Monitor progress."
            )
        elif prev_completed_duration >= self.rules.get("completed_duration_thresholds_min", {}).get("moderate", 175.0):
            risk_score += 10
            flags.append("MODERATE_PRIOR_COMPLETION_DELAY")
            advisories.append("ℹ️ Normal Pace: Driver completed their last delivery on schedule (~2.9 hours).")
            
        # 2. Evaluate Active In-Flight Workload (Wave Congestion)
        if active_inflight >= self.rules.get("active_inflight_thresholds", {}).get("heavy_wave", 15):
            risk_score += 25
            flags.append("HEAVY_INFLIGHT_WAVE")
            advisories.append(f"📦 Heavy Load: Driver is currently carrying {active_inflight} packages in this run.")
        elif active_inflight >= self.rules.get("active_inflight_thresholds", {}).get("moderate_wave", 8):
            risk_score += 10
            flags.append("MODERATE_INFLIGHT_WAVE")
            advisories.append(f"📦 Moderate Load: Driver is carrying {active_inflight} packages.")
            
        # 3. Evaluate Spatial Transition Distance
        if jump_dist_km >= self.rules.get("spatial_jump_thresholds_km", {}).get("high_friction", 2.5):
            risk_score += 20
            flags.append("HIGH_TRANSITION_DISTANCE")
            advisories.append(f"📍 Long Pickup Distance: Driver must travel {jump_dist_km:.1f} km from last pickup location.")
        elif jump_dist_km >= self.rules.get("spatial_jump_thresholds_km", {}).get("moderate_friction", 1.0):
            risk_score += 10
            flags.append("MODERATE_TRANSITION_DISTANCE")
            
        # 4. Long Distance Delivery Combined Context
        if current_distance_km > 5.0 and risk_score > 30:
            risk_score += 15
            flags.append("COMPOUNDED_DISTANCE_RISK")
            advisories.append("🚨 Long Route & Busy Driver: Long delivery distance (>5 km) combined with a busy driver schedule.")
            
        # Overall Risk Tier
        if risk_score >= 50:
            tier = "High Delay Risk"
            color = "#ef4444"
        elif risk_score >= 30:
            tier = "Elevated Delay Risk"
            color = "#f97316"
        elif risk_score >= 15:
            tier = "Moderate Risk"
            color = "#eab308"
        else:
            tier = "On Track (Low Risk)"
            color = "#22c55e"
            if not advisories:
                advisories.append("✅ All conditions are optimal. No elevated delay risks detected.")
                
        return {
            "risk_score": min(100, risk_score),
            "risk_tier": tier,
            "badge_color": color,
            "flags": flags,
            "advisories": advisories
        }
