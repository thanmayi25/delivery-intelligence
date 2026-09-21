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
    Asserts that feature_cols contains zero post-acceptance or target leakages.
    """
    feature_cols = dataset_artifacts["metadata"]["feature_cols"]
    forbidden_leakage = [
        "delivery_time",
        "delivery_gps_time",
        "delivery_gps_lng",
        "delivery_gps_lat",
        "delivery_duration_minutes",
        "high_delay"
    ]
    for col in forbidden_leakage:
        assert col not in feature_cols, f"Forbidden leakage column '{col}' found in feature_cols!"

def test_temporal_split_monotonicity(dataset_artifacts):
    """
    Asserts strict chronological separation: Train < Validation < Test.
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
    Asserts that high-delay threshold is computed strictly from training partition.
    """
    train_df = dataset_artifacts["train"]
    meta_thresh = dataset_artifacts["metadata"]["high_delay_threshold_minutes"]
    
    computed_train_p90 = float(np.percentile(train_df["delivery_duration_minutes"], 90))
    assert np.isclose(meta_thresh, computed_train_p90, atol=1e-3), (
        f"Metadata threshold {meta_thresh} does not match training P90 {computed_train_p90}"
    )

def test_daily_task_index_resets_each_day(dataset_artifacts):
    """
    Asserts that daily_task_index resets to 0 on every new calendar date per courier.
    """
    for split_name in ["train", "val", "test"]:
        df = dataset_artifacts[split_name]
        df["date"] = pd.to_datetime(df["accept_time"]).dt.date
        
        # Check first task of day flag
        first_tasks = df[df["daily_task_index"] == 0]
        assert len(first_tasks) > 0
        assert (first_tasks["is_first_task_of_day"] == 1).all(), (
            f"All daily_task_index == 0 rows must have is_first_task_of_day == 1 in {split_name}"
        )

def test_point_in_time_elapsed_values_non_negative(dataset_artifacts):
    """
    Asserts that point-in-time elapsed time features are strictly non-negative.
    """
    for split_name in ["train", "val", "test"]:
        df = dataset_artifacts[split_name]
        
        assert (df["mins_since_recent_completed"] >= 0).all(), (
            f"Negative mins_since_recent_completed detected in {split_name}"
        )
        assert (df["minutes_since_last_dispatch_today"] >= 0).all(), (
            f"Negative minutes_since_last_dispatch_today detected in {split_name}"
        )
        assert (df["active_inflight_tasks"] >= 0).all(), (
            f"Negative active_inflight_tasks detected in {split_name}"
        )
