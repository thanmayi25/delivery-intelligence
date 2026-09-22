from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "reports" / "delivery_cleaned_with_distance.parquet"

# Load feature computation module dynamically
spec = importlib.util.spec_from_file_location("ml_prep", PROJECT_ROOT / "scripts" / "03_ml_preprocessing.py")
ml_prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml_prep)

@pytest.fixture(scope="module")
def base_raw_data():
    raw_df = pd.read_parquet(DATA_RAW)
    raw_df = raw_df[
        raw_df["delivery_duration_minutes"].notna() & 
        (raw_df["delivery_duration_minutes"] >= 0) & 
        raw_df["delivery_distance_km"].notna()
    ].copy()
    for col in ["accept_time", "delivery_time"]:
        raw_df[col] = pd.to_datetime(raw_df[col])
    raw_df = raw_df.sort_values(["courier_id", "accept_time", "order_id"]).reset_index(drop=True)
    
    # Compute full dataset features (a)
    full_dfs = []
    for cid, group in raw_df.groupby("courier_id"):
        full_dfs.append(ml_prep.compute_courier_point_in_time_features(group))
    full_df = pd.concat(full_dfs, ignore_index=True).set_index("order_id")
    
    return {
        "raw_df": raw_df,
        "full_df": full_df
    }

@pytest.mark.parametrize("cutoff_str", [
    "1900-06-15 12:00:00",
    "1900-07-20 12:00:00",
    "1900-08-15 12:00:00"
])
def test_temporal_truncation_invariance(base_raw_data, cutoff_str):
    """
    TRUNCATION PROPERTY TEST:
    Simulates 'now = T' by:
    1. Dropping all tasks dispatched at or after T (accept_time >= T).
    2. Masking outcomes of tasks not yet finished at T (delivery_time >= T).
    
    For all tasks dispatched before T, the features generated on the full dataset (a)
    and the truncated dataset (b) MUST be mathematically identical. Any difference
    proves the feature used future information from after T.
    """
    raw_df = base_raw_data["raw_df"]
    full_df = base_raw_data["full_df"]
    T = pd.Timestamp(cutoff_str)
    
    # Dataset (b): tasks dispatched strictly before T
    trunc_raw = raw_df[raw_df["accept_time"] < T].copy()
    
    # Mask unobserved outcomes at time T (delivery finishes >= T)
    mask_future = trunc_raw["delivery_time"] >= T
    # Set to future sentinel so in-flight status is preserved while future duration is inaccessible
    trunc_raw.loc[mask_future, "delivery_time"] = T + pd.Timedelta(days=100)
    trunc_raw.loc[mask_future, "delivery_duration_minutes"] = np.nan
    
    trunc_dfs = []
    for cid, group in trunc_raw.groupby("courier_id"):
        trunc_dfs.append(ml_prep.compute_courier_point_in_time_features(group))
    trunc_df = pd.concat(trunc_dfs, ignore_index=True).set_index("order_id")
    
    feature_cols_to_verify = [
        "duration_of_most_recently_completed_task",
        "mins_since_recent_completed",
        "has_prior_completed_task",
        "active_inflight_tasks",
        "daily_task_index",
        "is_first_task_of_day",
        "minutes_since_last_dispatch_today",
        "previous_accept_distance_km",
        "tasks_previous_1h",
        "tasks_previous_3h"
    ]
    
    for col in feature_cols_to_verify:
        full_vals = full_df.loc[trunc_df.index, col]
        trunc_vals = trunc_df[col]
        diff_mask = ~np.isclose(full_vals.fillna(-999.0), trunc_vals.fillna(-999.0), atol=1e-5)
        
        num_diff = diff_mask.sum()
        assert num_diff == 0, (
            f"TRUNCATION LEAKAGE DETECTED at cutoff {T} for feature '{col}'! "
            f"{num_diff} rows differed when future data after {T} was removed. "
            f"First mismatch order_id: {trunc_df.index[diff_mask][0]}"
        )
