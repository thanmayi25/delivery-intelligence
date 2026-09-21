# Delivery Intelligence Deployment & Serving Guide

This document outlines the two-tier architecture for deploying the Delivery Intelligence system:
- **Phase 1**: Interactive Research & Operations Application (Streamlit)
- **Phase 2**: Production REST Microservice (FastAPI + Uvicorn)

---

## 🏛️ Architecture Overview

```text
 ┌──────────────────────────────────────────────────────────────┐
 │                      Client Layer                            │
 │  ┌────────────────────────────┐  ┌────────────────────────┐  │
 │  │ Streamlit Dashboard (:8501)│  │ Logistics ERP / App    │  │
 │  └─────────────┬──────────────┘  └───────────┬────────────┘  │
 └────────────────┼─────────────────────────────┼───────────────┘
                  │                             │
                  ▼                             ▼
 ┌──────────────────────────────────────────────────────────────┐
 │             Phase 2: FastAPI Microservice (:8000)            │
 │  • POST /predict/dispatch-evaluation                         │
 │  • POST /predict/raw-gps                                     │
 │  • POST /predict/batch                                       │
 │  • GET  /health & /models/rules                              │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                     ML & Advisory Engine                     │
 │  • Champion Regressor: LightGBM (MAE: 74.59m | R²: 0.4048)   │
 │  • Champion Classifier: LightGBM (ROC-AUC: 0.8699)           │
 │  • Sequence Advisory Engine (Dynamic Ripple Rules)           │
 └──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Running Locally

### Option A: Launch Phase 1 (Streamlit Research Dashboard)
```powershell
streamlit run app/delivery_intelligence_app.py
```
* **URL**: `http://localhost:8501`
* **Features**: Live dispatch simulator, generic dataset CSV/Parquet uploader with auto-feature generation, model benchmarks, and empirical sequence matrices.

---

### Option B: Launch Phase 2 (FastAPI Production Backend)
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Base URL**: `http://localhost:8000`
* **Interactive Swagger UI**: `http://localhost:8000/docs`
* **Redoc Specification**: `http://localhost:8000/redoc`

---

## 🐳 Containerized Deployment (Docker & Docker Compose)

### 1. Build and Run Both Services
```bash
docker-compose up --build -d
```

### 2. Verify Running Containers
```bash
docker-compose ps
```
- FastAPI API will be available at `http://localhost:8000/docs`
- Streamlit UI will be available at `http://localhost:8501`

### 3. Stop Containers
```bash
docker-compose down
```

---

## 📡 API Endpoint Reference

### 1. `POST /predict/dispatch-evaluation`
**Request Payload**:
```json
{
  "delivery_distance_km": 3.2,
  "tasks_previous_1h": 4,
  "tasks_previous_3h": 10,
  "previous_tasks": 5,
  "task_gap_minutes": 15.0,
  "previous_delivery_duration": 140.0,
  "previous_to_current_distance_km": 0.8,
  "is_first_task_of_courier": 0,
  "accept_hour": 14,
  "accept_minute": 30,
  "accept_weekday": 2,
  "is_weekend": 0,
  "time_period": "Afternoon"
}
```

**Response**:
```json
{
  "predicted_duration_minutes": 171.5,
  "predicted_duration_hours": 2.86,
  "high_delay_risk_probability": 0.0753,
  "is_high_delay_flag": false,
  "risk_level": "LOW",
  "sequence_advisory": {
    "risk_score": 10,
    "risk_tier": "LOW RISK / OPTIMAL",
    "color": "#22c55e",
    "flags": [
      "MODERATE_PRIOR_DELAY"
    ],
    "advisories": [
      "ℹ️ Moderate preceding duration (>2h). Monitor upcoming order progress."
    ]
  },
  "metadata": {
    "model_engine": "LightGBM",
    "high_delay_threshold_minutes": 384.0,
    "version": "1.0.0"
  }
}
```

---

### 2. `POST /predict/raw-gps`
Evaluates dispatch directly from pickup/dropoff coordinates and timestamp:
```json
{
  "order_id": "ORD-9021",
  "courier_id": "CR-44",
  "accept_gps_lng": 126.5669,
  "accept_gps_lat": 43.8120,
  "delivery_gps_lng": 126.5445,
  "delivery_gps_lat": 43.8502,
  "accept_time_str": "2026-09-20 14:30:00",
  "tasks_previous_1h": 3,
  "tasks_previous_3h": 8,
  "previous_delivery_duration": 120.0,
  "previous_to_current_distance_km": 0.8
}
```
