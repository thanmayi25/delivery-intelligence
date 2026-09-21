from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class OrderFeatureInput(BaseModel):
    delivery_distance_km: float = Field(..., ge=0.0, le=200.0, description="Haversine distance between pickup and dropoff in km", example=2.8)
    tasks_previous_1h: int = Field(default=2, ge=0, description="Tasks accepted by courier in prior 1 hour", example=2)
    tasks_previous_3h: int = Field(default=6, ge=0, description="Tasks accepted by courier in prior 3 hours", example=6)
    active_inflight_tasks: int = Field(default=5, ge=0, description="Active uncompleted parcels currently in flight for this courier", example=5)
    daily_task_index: int = Field(default=3, ge=0, description="0-indexed sequence count of tasks for this courier today", example=3)
    is_first_task_of_day: int = Field(default=0, ge=0, le=1, description="1 if first dispatch of today's shift, 0 otherwise", example=0)
    minutes_since_last_dispatch_today: float = Field(default=0.0, ge=0.0, description="Minutes elapsed since last accepted dispatch today", example=0.0)
    duration_of_most_recently_completed_task: float = Field(default=175.0, ge=0.0, description="Duration in minutes of courier's most recently delivered parcel", example=175.0)
    has_prior_completed_task: int = Field(default=1, ge=0, le=1, description="1 if courier has completed at least 1 delivery before now, 0 otherwise", example=1)
    mins_since_recent_completed: float = Field(default=30.0, ge=0.0, description="Minutes elapsed since the courier's most recent parcel delivery", example=30.0)
    previous_accept_distance_km: float = Field(default=0.5, ge=0.0, description="Distance from prior pickup location in km", example=0.5)
    accept_hour: int = Field(default=9, ge=0, le=23, description="Acceptance hour (0-23)", example=9)
    accept_minute: int = Field(default=15, ge=0, le=59, description="Acceptance minute (0-59)", example=15)
    accept_weekday: int = Field(default=2, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)", example=2)
    is_weekend: int = Field(default=0, ge=0, le=1, description="1 if weekend, 0 if weekday", example=0)
    time_period: str = Field(default="Morning", description="Time period: Morning, Afternoon, Evening, Night", example="Morning")

class RawGPSOrderInput(BaseModel):
    order_id: Optional[str] = Field(default="ORD-1001", description="Unique identifier for the order")
    courier_id: Optional[str] = Field(default="CR-502", description="Courier identifier")
    accept_gps_lng: float = Field(..., description="Order pickup longitude", example=126.5669)
    accept_gps_lat: float = Field(..., description="Order pickup latitude", example=43.8120)
    delivery_gps_lng: float = Field(..., description="Order dropoff longitude", example=126.5445)
    delivery_gps_lat: float = Field(..., description="Order dropoff latitude", example=43.8502)
    accept_time_str: str = Field(..., description="Local timestamp format 'YYYY-MM-DD HH:MM:SS' or 'MM-DD HH:MM:SS'", example="2026-09-20 09:30:00")
    active_inflight_tasks: Optional[int] = Field(default=5, description="Active parcels currently in flight")
    tasks_previous_1h: Optional[int] = Field(default=2, description="Courier workload in past 1 hour")
    tasks_previous_3h: Optional[int] = Field(default=6, description="Courier workload in past 3 hours")
    duration_of_most_recently_completed_task: Optional[float] = Field(default=175.0, description="Duration of previous completed delivery in min")
    previous_accept_distance_km: Optional[float] = Field(default=0.5, description="Distance from previous pickup location")

class PredictionInterval(BaseModel):
    p10_minutes: float = Field(..., description="10th percentile lower bound (minutes)")
    p50_median_minutes: float = Field(..., description="50th percentile median estimate (minutes)")
    p90_minutes: float = Field(..., description="90th percentile upper bound (minutes)")
    interval_coverage_pct: float = Field(default=80.0, description="Expected interval coverage (80%)")

class SequenceAdvisoryOutput(BaseModel):
    risk_score: int = Field(..., description="Calculated sequence friction risk score (0-100)")
    risk_tier: str = Field(..., description="Risk tier: LOW RISK / OPTIMAL, MODERATE RISK, HIGH RISK, CRITICAL RISK")
    badge_color: str = Field(..., description="Hex color code for UI badge")
    flags: List[str] = Field(..., description="Triggered risk rule flags")
    advisories: List[str] = Field(..., description="Actionable dispatch directives and operational recommendations")

class DispatchEvaluationResponse(BaseModel):
    predicted_duration_minutes: float = Field(..., description="Point prediction of delivery duration (ETA) in minutes")
    predicted_duration_hours: float = Field(..., description="Point prediction in hours")
    prediction_interval: PredictionInterval
    high_delay_risk_probability: float = Field(..., description="Calibrated probability of exceeding high-delay threshold (380 min)")
    is_high_delay_flag: bool = Field(..., description="True if calibrated probability exceeds optimal decision threshold")
    risk_level: str = Field(..., description="Risk categorization: LOW, ELEVATED, CRITICAL")
    sequence_advisory: SequenceAdvisoryOutput
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Model version, threshold, and execution metadata")

class BatchDispatchEvaluationRequest(BaseModel):
    orders: List[OrderFeatureInput] = Field(..., description="List of orders to evaluate in batch")

class BatchDispatchEvaluationResponse(BaseModel):
    total_orders: int
    average_predicted_duration_minutes: float
    high_delay_count: int
    high_delay_rate_percent: float
    results: List[DispatchEvaluationResponse]

class HealthResponse(BaseModel):
    status: str
    service_name: str
    version: str
    regression_model_loaded: bool
    quantile_models_loaded: bool
    classification_model_loaded: bool
    sequence_rules_loaded: bool
