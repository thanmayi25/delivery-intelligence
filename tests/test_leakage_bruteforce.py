from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "reports" / "delivery_cleaned_with_distance.parquet"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

def haversine_reference(lon1, lat1, lon2, lat2):
    """
    Completely independent reference implementation of Haversine distance.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c

@pytest.fixture(scope="module")
def sampled_ground_truth_dataset():
    """
    Loads raw un-preprocessed data and samples >= 500 rows with forced inclusion of:
    1. First task per courier (cold start)
    2. First task per day per courier
    3. Tasks where naive shift(1) prior task was still UNFINISHED at dispatch time
    4. Tasks with tied accept_time timestamps (batch dispatches)
    5. Early dataset tasks (first 3 calendar days)
    6. Random stratified sample across dataset
    """
    raw_df = pd.read_parquet(DATA_RAW)
    raw_df = raw_df[raw_df["delivery_duration_minutes"].notna() & (raw_df["delivery_duration_minutes"] >= 0) & raw_df["delivery_distance_km"].notna()].copy()
    for col in ["accept_time", "delivery_time"]:
        raw_df[col] = pd.to_datetime(raw_df[col])
        
    raw_df = raw_df.sort_values(["courier_id", "accept_time", "order_id"]).reset_index(drop=True)
    raw_df["date"] = raw_df["accept_time"].dt.date
    
    # Identify edge cases
    raw_df["prev_deliv"] = raw_df.groupby("courier_id")["delivery_time"].shift(1)
    raw_df["naive_unfin"] = raw_df["prev_deliv"] > raw_df["accept_time"]
    raw_df["tied"] = raw_df.duplicated(subset=["courier_id", "accept_time"], keep=False)
    raw_df["first_courier"] = ~raw_df.duplicated(subset=["courier_id"], keep="first")
    raw_df["first_day"] = ~raw_df.duplicated(subset=["courier_id", "date"], keep="first")
    raw_df["early"] = raw_df["date"] <= raw_df["date"].min() + pd.Timedelta(days=3)
    
    # Identify boundary equality cases: delivery_time == accept_time
    boundary_ids = []
    for cid, group in raw_df.groupby("courier_id"):
        acc = group["accept_time"].to_numpy()
        deliv = group["delivery_time"].to_numpy()
        oids = group["order_id"].to_numpy()
        for i in range(len(group)):
            if np.any(deliv[:i] == acc[i]):
                boundary_ids.append(oids[i])
                
    np.random.seed(42)
    sample_ids = set()
    
    # Force inclusion of all boundary equality cases (< vs <= boundary)
    sample_ids.update(boundary_ids)
    # 1. All first courier tasks
    sample_ids.update(raw_df[raw_df["first_courier"]]["order_id"].tolist())
    # 2. Sample first task of day
    sample_ids.update(raw_df[raw_df["first_day"]].sample(n=100, random_state=42)["order_id"].tolist())
    # 3. Sample naive unfinished
    sample_ids.update(raw_df[raw_df["naive_unfin"]].sample(n=150, random_state=42)["order_id"].tolist())
    # 4. Sample tied timestamps
    sample_ids.update(raw_df[raw_df["tied"]].sample(n=150, random_state=42)["order_id"].tolist())
    # 5. Early tasks
    sample_ids.update(raw_df[raw_df["early"]].sample(n=min(50, raw_df["early"].sum()), random_state=42)["order_id"].tolist())
    # 6. General random sample
    remaining = raw_df[~raw_df["order_id"].isin(sample_ids)]
    sample_ids.update(remaining.sample(n=150, random_state=42)["order_id"].tolist())
    
    sample_df = raw_df[raw_df["order_id"].isin(sample_ids)].copy()
    
    # Load production processed tables
    train_df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    val_df = pd.read_parquet(DATA_PROCESSED / "val_features.parquet")
    test_df = pd.read_parquet(DATA_PROCESSED / "test_features.parquet")
    all_prod = pd.concat([train_df, val_df, test_df], ignore_index=True).set_index("order_id")
    
    with open(DATA_PROCESSED / "metadata.json", "r") as f:
        metadata = json.load(f)
        
    return {
        "raw_df": raw_df,
        "sample_df": sample_df,
        "all_prod": all_prod,
        "impute_values": metadata["impute_values"]
    }

def test_independent_bruteforce_point_in_time_features(sampled_ground_truth_dataset):
    """
    CRITICAL TEST: Independently recomputes all 10 point-in-time sequential features by hand
    from raw data for >= 500 sampled edge-case rows and asserts exact equality against the pipeline.
    """
    raw_df = sampled_ground_truth_dataset["raw_df"]
    sample_df = sampled_ground_truth_dataset["sample_df"]
    all_prod = sampled_ground_truth_dataset["all_prod"]
    impute_values = sampled_ground_truth_dataset["impute_values"]
    
    assert len(sample_df) >= 500, f"Sample size must be >= 500, got {len(sample_df)}"
    
    for idx, row in sample_df.iterrows():
        oid = row["order_id"]
        cid = row["courier_id"]
        t_acc = row["accept_time"]
        d_curr = row["date"]
        lng_curr = row["accept_gps_lng"]
        lat_curr = row["accept_gps_lat"]
        
        prod_row = all_prod.loc[oid]
        
        # 1. Plain, brute-force filter: tasks for same courier dispatched strictly before this task
        courier_tasks = raw_df[raw_df["courier_id"] == cid]
        prior_tasks = courier_tasks[
            (courier_tasks["accept_time"] < t_acc) | 
            ((courier_tasks["accept_time"] == t_acc) & (courier_tasks.index < idx))
        ]
        
        # 2. Strict completed task filter: delivery_time <= t_acc
        completed = prior_tasks[prior_tasks["delivery_time"] <= t_acc]
        
        # Independent feature calculations:
        # A. Most recently completed task duration & elapsed minutes
        if len(completed) > 0:
            max_deliv_time = completed["delivery_time"].max()
            most_recent_task = completed[completed["delivery_time"] == max_deliv_time].iloc[-1]
            exp_dur = float(most_recent_task["delivery_duration_minutes"])
            exp_mins_since_comp = (t_acc - max_deliv_time).total_seconds() / 60.0
            exp_has_prior = 1
        else:
            exp_dur = impute_values["duration_of_most_recently_completed_task"]
            exp_mins_since_comp = impute_values["mins_since_recent_completed"]
            exp_has_prior = 0
            
        # B. Active in-flight tasks
        exp_inflight = int((prior_tasks["delivery_time"] > t_acc).sum())
        
        # C. Daily task index & first task of day
        prior_today = prior_tasks[prior_tasks["date"] == d_curr]
        exp_daily_task_index = len(prior_today)
        exp_is_first = 1 if exp_daily_task_index == 0 else 0
        
        # D. Time & distance since last dispatch today
        if len(prior_today) > 0:
            prev_row = prior_today.iloc[-1]
            exp_mins_dispatch = max(0.0, (t_acc - prev_row["accept_time"]).total_seconds() / 60.0)
            exp_prev_dist = haversine_reference(prev_row["accept_gps_lng"], prev_row["accept_gps_lat"], lng_curr, lat_curr)
        else:
            exp_mins_dispatch = impute_values["minutes_since_last_dispatch_today"]
            exp_prev_dist = impute_values["previous_accept_distance_km"]
            
        # E. Rolling 1h / 3h workload counts (strictly [t_acc - W, t_acc))
        exp_tasks_1h = int(((courier_tasks["accept_time"] >= t_acc - pd.Timedelta(hours=1)) & 
                            (courier_tasks["accept_time"] < t_acc)).sum())
        exp_tasks_3h = int(((courier_tasks["accept_time"] >= t_acc - pd.Timedelta(hours=3)) & 
                            (courier_tasks["accept_time"] < t_acc)).sum())
                            
        # Assertions with descriptive debug info
        checks = [
            ("has_prior_completed_task", prod_row["has_prior_completed_task"], exp_has_prior),
            ("duration_of_most_recently_completed_task", prod_row["duration_of_most_recently_completed_task"], exp_dur),
            ("mins_since_recent_completed", prod_row["mins_since_recent_completed"], exp_mins_since_comp),
            ("active_inflight_tasks", prod_row["active_inflight_tasks"], exp_inflight),
            ("daily_task_index", prod_row["daily_task_index"], exp_daily_task_index),
            ("is_first_task_of_day", prod_row["is_first_task_of_day"], exp_is_first),
            ("minutes_since_last_dispatch_today", prod_row["minutes_since_last_dispatch_today"], exp_mins_dispatch),
            ("previous_accept_distance_km", prod_row["previous_accept_distance_km"], exp_prev_dist),
            ("tasks_previous_1h", prod_row["tasks_previous_1h"], exp_tasks_1h),
            ("tasks_previous_3h", prod_row["tasks_previous_3h"], exp_tasks_3h),
        ]
        
        for fname, pval, eval in checks:
            assert np.isclose(pval, eval, atol=1e-5), (
                f"LEAKAGE / MISMATCH DETECTED for order_id={oid} (courier_id={cid}, accept_time={t_acc}): "
                f"Feature '{fname}' -> Pipeline Value: {pval}, Expected Value: {eval}"
            )
