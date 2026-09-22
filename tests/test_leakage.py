from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

@pytest.fixture(scope="module")
def dataset_artifacts():
    train_df = pd.read_parquet(DATA_DIR / "train_features.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val_features.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_features.parquet")
    
    with open(DATA_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)
        
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "metadata": metadata
    }

def test_feature_list_excludes_future_targets(dataset_artifacts):
    """
    SMOKE TEST: Asserts that feature_cols contains zero outcome, post-dispatch, or target columns.
    """
    feature_cols = dataset_artifacts["metadata"]["feature_cols"]
    forbidden_leakage = [
        "delivery_time",
        "delivery_gps_time",
        "delivery_gps_lng",
        "delivery_gps_lat",
        "delivery_duration_minutes",
        "high_delay",
        "completion_time",
        "end_time",
        "actual_duration",
        "pickup_time",
        "delay_flag",
        "is_delayed",
        "delivery_status"
    ]
    for col in forbidden_leakage:
        assert col not in feature_cols, f"Forbidden leakage column '{col}' found in feature_cols!"

def test_temporal_split_monotonicity(dataset_artifacts):
    """
    Asserts strict chronological separation across temporal splits: Train < Validation < Test.
    """
    train_df = dataset_artifacts["train"]
    val_df = dataset_artifacts["val"]
    test_df = dataset_artifacts["test"]
    
    train_max = pd.to_datetime(train_df["accept_time"]).max()
    val_min = pd.to_datetime(val_df["accept_time"]).min()
    val_max = pd.to_datetime(val_df["accept_time"]).max()
    test_min = pd.to_datetime(test_df["accept_time"]).min()
    
    assert train_max <= val_min, f"Train max ({train_max}) must be <= Val min ({val_min})"
    assert val_max <= test_min, f"Val max ({val_max}) must be <= Test min ({test_min})"

def test_high_delay_threshold_strictly_fitted_on_train(dataset_artifacts):
    """
    Asserts that high-delay threshold ($P_{90}$) is computed strictly from training partition.
    """
    train_df = dataset_artifacts["train"]
    meta_thresh = dataset_artifacts["metadata"]["high_delay_threshold_minutes"]
    
    computed_train_p90 = float(np.percentile(train_df["delivery_duration_minutes"], 90))
    assert np.isclose(meta_thresh, computed_train_p90, atol=1e-3), (
        f"Metadata threshold {meta_thresh} does not match training P90 {computed_train_p90}"
    )

def test_daily_task_index_resets_each_day_independent(dataset_artifacts):
    """
    Independently verifies daily_task_index by grouping by (courier_id, calendar_date)
    across the entire chronological dataset and comparing the cumulative rank.
    """
    all_df = pd.concat([
        dataset_artifacts["train"],
        dataset_artifacts["val"],
        dataset_artifacts["test"]
    ], ignore_index=True)
    
    all_df["date"] = pd.to_datetime(all_df["accept_time"]).dt.date
    all_df = all_df.sort_values(["courier_id", "accept_time", "order_id"]).reset_index(drop=True)
    
    # Independent rank computation
    all_df["expected_daily_index"] = all_df.groupby(["courier_id", "date"]).cumcount()
    all_df["expected_is_first"] = (all_df["expected_daily_index"] == 0).astype(int)
    
    mismatches_idx = (all_df["daily_task_index"] != all_df["expected_daily_index"]).sum()
    mismatches_first = (all_df["is_first_task_of_day"] != all_df["expected_is_first"]).sum()
    
    assert mismatches_idx == 0, f"Found {mismatches_idx} daily_task_index mismatches across dataset"
    assert mismatches_first == 0, f"Found {mismatches_first} is_first_task_of_day mismatches across dataset"

def test_completed_history_contains_zero_future_deliveries(dataset_artifacts):
    """
    Asserts that whenever has_prior_completed_task == 1, the completed task lookup
    could not have come from a task completed after accept_time.
    """
    for split_name in ["train", "val", "test"]:
        df = dataset_artifacts[split_name]
        
        # mins_since_recent_completed must be >= 0 (strictly in past)
        valid_history = df[df["has_prior_completed_task"] == 1]
        assert (valid_history["mins_since_recent_completed"] >= 0).all(), (
            f"Future completion detected in {split_name} (negative mins_since_recent_completed)"
        )
        assert (df["active_inflight_tasks"] >= 0).all(), (
            f"Negative active_inflight_tasks detected in {split_name}"
        )
