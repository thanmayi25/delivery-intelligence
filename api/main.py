import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    OrderFeatureInput,
    RawGPSOrderInput,
    DispatchEvaluationResponse,
    BatchDispatchEvaluationRequest,
    BatchDispatchEvaluationResponse,
    HealthResponse
)
from api.service import service_instance

app = FastAPI(
    title="Delivery Intelligence & Sequence Analytics API",
    description="""
Production-grade REST API for express logistics, delivery duration estimation (ETA),
prediction intervals ([P10, P90]), high-delay risk classification, and task-sequence advisory.

### Core Capabilities:
* **Duration ETA Prediction & Quantile Intervals**: Point predictions and [P10, P90] 80% uncertainty bounds.
* **Calibrated High-Delay Risk Modeling**: Isotonic/Platt calibrated probabilities of exceeding high-delay thresholds.
* **Point-in-Time Sequence Advisory**: Dynamic evaluation of prior completed task latency and spatial transitions.
* **Batch Dispatch Analytics**: Multi-order evaluation endpoint.
""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time-Seconds"] = f"{process_time:.4f}"
    return response

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Diagnostics"],
    summary="Health check and model status"
)
async def health_check():
    """Returns the operational status and loaded model artifacts."""
    return HealthResponse(
        status="healthy",
        service_name="Delivery Intelligence API",
        version="2.0.0",
        regression_model_loaded=service_instance.reg_model is not None,
        quantile_models_loaded=bool(service_instance.quantile_models),
        classification_model_loaded=service_instance.clf_model is not None,
        sequence_rules_loaded=bool(service_instance.rules)
    )

@app.post(
    "/predict/dispatch-evaluation",
    response_model=DispatchEvaluationResponse,
    tags=["Predictive Intelligence"],
    summary="Evaluate full dispatch scenario (ETA + Intervals + Delay Risk + Sequence Advisory)"
)
async def evaluate_dispatch(order: OrderFeatureInput):
    try:
        return service_instance.evaluate_single_order(order)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post(
    "/predict/raw-gps",
    response_model=DispatchEvaluationResponse,
    tags=["Predictive Intelligence"],
    summary="Evaluate dispatch from raw GPS coordinates and timestamp"
)
async def evaluate_raw_gps(raw_order: RawGPSOrderInput):
    try:
        return service_instance.evaluate_raw_gps_order(raw_order)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coordinate processing/inference error: {str(e)}"
        )

@app.post(
    "/predict/batch",
    response_model=BatchDispatchEvaluationResponse,
    tags=["Batch Operations"],
    summary="Evaluate a batch of delivery orders"
)
async def evaluate_batch_dispatch(payload: BatchDispatchEvaluationRequest):
    try:
        return service_instance.evaluate_batch(payload.orders)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )

@app.get(
    "/models/rules",
    tags=["Model Interpretability"],
    summary="Inspect task-sequence advisory rules and empirical benchmarks"
)
async def get_sequence_rules():
    return service_instance.rules

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
