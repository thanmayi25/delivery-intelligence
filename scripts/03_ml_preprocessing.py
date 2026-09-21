from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PARQUET = PROJECT_ROOT / "reports" / "delivery_cleaned_with_distance.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def haversine_distance_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c

def assign_time_period(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

def compute_courier_point_in_time_features(group):
    """
    Computes strict point-in-time features for a single courier's chronologically sorted orders.
    Guarantees ZERO access to any event or delivery completed after the current order's accept_time.
    """
    group = group.sort_values(["accept_time", "order_id"]).copy()
    accept_times = group["accept_time"].to_numpy()
    delivery_times = group["delivery_time"].to_numpy()
    durations = group["delivery_duration_minutes"].to_numpy()
    accept_lngs = group["accept_gps_lng"].to_numpy()
    accept_lats = group["accept_gps_lat"].to_numpy()
    
    # Extract date for daily reset
    dates = group["accept_time"].dt.date.to_numpy()
    
    n = len(group)
    recent_completed_dur = np.full(n, np.nan)
    mins_since_recent_completed = np.full(n, np.nan)
    has_prior_completed = np.zeros(n, dtype=int)
    active_inflight = np.zeros(n, dtype=int)
    
    daily_task_index = np.zeros(n, dtype=int)
    is_first_task_of_day = np.zeros(n, dtype=int)
    mins_since_last_dispatch_today = np.full(n, np.nan)
    prev_accept_dist_km = np.full(n, np.nan)
    
    curr_day = None
    day_task_cnt = 0
    prev_accept_time_today = None
    prev_lng_today = None
    prev_lat_today = None
    
    for i in range(n):
        t_accept = accept_times[i]
        d_today = dates[i]
        
        # 1. Daily shift tracking & resets
        if curr_day != d_today:
            curr_day = d_today
            day_task_cnt = 0
            daily_task_index[i] = 0
            is_first_task_of_day[i] = 1
            mins_since_last_dispatch_today[i] = np.nan
            prev_accept_dist_km[i] = np.nan
        else:
            day_task_cnt += 1
            daily_task_index[i] = day_task_cnt
            is_first_task_of_day[i] = 0
            if prev_accept_time_today is not None:
                mins_diff = (t_accept - prev_accept_time_today).astype('timedelta64[s]').astype(float) / 60.0
                mins_since_last_dispatch_today[i] = max(0.0, mins_diff)
            if prev_lng_today is not None and prev_lat_today is not None:
                prev_accept_dist_km[i] = haversine_distance_km(
                    prev_lng_today, prev_lat_today, accept_lngs[i], accept_lats[i]
                )
                
        prev_accept_time_today = t_accept
        prev_lng_today = accept_lngs[i]
        prev_lat_today = accept_lats[i]
        
        # 2. Strict point-in-time completed task lookup
        # Completed tasks j < i where delivery_times[j] <= t_accept
        completed_mask = delivery_times[:i] <= t_accept
        if np.any(completed_mask):
            comp_indices = np.where(completed_mask)[0]
            comp_deliv_times = delivery_times[comp_indices]
            best_idx = comp_indices[np.argmax(comp_deliv_times)]
            recent_completed_dur[i] = durations[best_idx]
            mins_since_recent_completed[i] = max(0.0, (t_accept - comp_deliv_times.max()).astype('timedelta64[s]').astype(float) / 60.0)
            has_prior_completed[i] = 1
            
        # 3. Active in-flight tasks as of dispatch
        active_inflight[i] = np.sum(delivery_times[:i] > t_accept)
        
    group["duration_of_most_recently_completed_task"] = recent_completed_dur
    group["mins_since_recent_completed"] = mins_since_recent_completed
    group["has_prior_completed_task"] = has_prior_completed
    group["active_inflight_tasks"] = active_inflight
    group["daily_task_index"] = daily_task_index
    group["is_first_task_of_day"] = is_first_task_of_day
    group["minutes_since_last_dispatch_today"] = mins_since_last_dispatch_today
    group["previous_accept_distance_km"] = prev_accept_dist_km
    
    # 4. Rolling 1h and 3h workload strictly before t_accept
    t_idx = group.set_index("accept_time")
    group["tasks_previous_1h"] = t_idx["order_id"].rolling("1h", closed="left").count().to_numpy()
    group["tasks_previous_3h"] = t_idx["order_id"].rolling("3h", closed="left").count().to_numpy()
    
    return group

def run_preprocessing():
    print("=" * 75)
    print("STEP 1: INGESTING AND VALIDATING RAW CLEANED DATASET")
    print("=" * 75)
    
    if not DATA_PARQUET.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PARQUET}")
        
    df = pd.read_parquet(DATA_PARQUET)
    print(f"Raw rows in cleaned parquet: {len(df)}")
    
    # Filter valid delivery durations and distances
    df = df[df["delivery_duration_minutes"].notna()].copy()
    df = df[df["delivery_duration_minutes"] >= 0].copy()
    df = df[df["delivery_distance_km"].notna()].copy()
    print(f"Rows after filtering invalid records: {len(df)}")
    
    for col in ["accept_time", "delivery_time"]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
    # Sort chronologically per courier
    df = df.sort_values(["courier_id", "accept_time", "order_id"]).reset_index(drop=True)
    
    print("\n" + "=" * 75)
    print("STEP 2: COMPUTING STRICT POINT-IN-TIME SEQUENTIAL FEATURES")
    print("=" * 75)
    
    # Apply point-in-time calculation per courier
    processed_courier_dfs = []
    for cid, group in df.groupby("courier_id"):
        processed_courier_dfs.append(compute_courier_point_in_time_features(group))
    df = pd.concat(processed_courier_dfs, ignore_index=True)
    
    df["tasks_previous_1h"] = df["tasks_previous_1h"].fillna(0).astype(int)
    df["tasks_previous_3h"] = df["tasks_previous_3h"].fillna(0).astype(int)
    
    # Time features derived from accept_time
    df["accept_hour"] = df["accept_time"].dt.hour
    df["accept_minute"] = df["accept_time"].dt.minute
    df["accept_weekday"] = df["accept_time"].dt.dayofweek
    df["is_weekend"] = (df["accept_weekday"] >= 5).astype(int)
    df["time_period"] = df["accept_hour"].apply(assign_time_period)
    
    feature_cols = [
        "delivery_distance_km",
        "tasks_previous_1h",
        "tasks_previous_3h",
        "active_inflight_tasks",
        "daily_task_index",
        "is_first_task_of_day",
        "minutes_since_last_dispatch_today",
        "duration_of_most_recently_completed_task",
        "has_prior_completed_task",
        "mins_since_recent_completed",
        "previous_accept_distance_km",
        "accept_hour",
        "accept_minute",
        "accept_weekday",
        "is_weekend",
        "time_period"
    ]
    
    target_cols = [
        "delivery_duration_minutes",
        "high_delay"
    ]
    
    meta_cols = [
        "order_id",
        "courier_id",
        "region_id",
        "city",
        "accept_time",
        "delivery_time"
    ]
    
    print("\n" + "=" * 75)
    print("STEP 3: DEFINING CHRONOLOGICAL SPLITS & FITTING TARGET THRESHOLD")
    print("=" * 75)
    
    # Sort globally by accept_time for chronological split
    df = df.sort_values(["accept_time", "order_id"]).reset_index(drop=True)
    
    total_n = len(df)
    train_n = int(0.70 * total_n)
    val_n = int(0.15 * total_n)
    
    train_df = df.iloc[:train_n].copy().reset_index(drop=True)
    val_df = df.iloc[train_n:train_n + val_n].copy().reset_index(drop=True)
    test_df = df.iloc[train_n + val_n:].copy().reset_index(drop=True)
    
    # FIT High-Delay Threshold STRICTLY on Training Partition ($P_{90}$)
    high_delay_threshold = float(np.percentile(train_df["delivery_duration_minutes"], 90))
    print(f"Empirical 90th percentile high-delay threshold (fit on Train): {high_delay_threshold:.2f} minutes ({high_delay_threshold/60.0:.2f} hours)")
    
    # Assign high_delay target to all partitions
    train_df["high_delay"] = (train_df["delivery_duration_minutes"] > high_delay_threshold).astype(int)
    val_df["high_delay"] = (val_df["delivery_duration_minutes"] > high_delay_threshold).astype(int)
    test_df["high_delay"] = (test_df["delivery_duration_minutes"] > high_delay_threshold).astype(int)
    df["high_delay"] = (df["delivery_duration_minutes"] > high_delay_threshold).astype(int)
    
    # Compute imputation medians strictly on Training partition
    impute_values = {
        "duration_of_most_recently_completed_task": float(train_df["duration_of_most_recently_completed_task"].median()),
        "mins_since_recent_completed": float(train_df["mins_since_recent_completed"].median()),
        "minutes_since_last_dispatch_today": float(train_df["minutes_since_last_dispatch_today"].median()),
        "previous_accept_distance_km": float(train_df["previous_accept_distance_km"].median())
    }
    
    print("\nTraining-derived imputation medians for first-task / missing sequences:")
    for k, v in impute_values.items():
        print(f" - {k}: {v:.2f}")
        
    for split_df in [train_df, val_df, test_df]:
        for col, val in impute_values.items():
            split_df[col] = split_df[col].fillna(val)
            
    print(f"\nTemporal Split Breakdown:")
    print(f" - Train (70%): {len(train_df)} rows | {train_df['accept_time'].min()} to {train_df['accept_time'].max()} | High-Delay: {train_df['high_delay'].mean()*100:.2f}%")
    print(f" - Val   (15%): {len(val_df)} rows | {val_df['accept_time'].min()} to {val_df['accept_time'].max()} | High-Delay: {val_df['high_delay'].mean()*100:.2f}%")
    print(f" - Test  (15%): {len(test_df)} rows | {test_df['accept_time'].min()} to {test_df['accept_time'].max()} | High-Delay: {test_df['high_delay'].mean()*100:.2f}%")
    
    print("\n" + "=" * 75)
    print("STEP 4: COURIER-GROUPED AND RANDOM SPLITS FOR GENERALIZATION BENCHMARK")
    print("=" * 75)
    
    # Courier Grouped Split (45 train couriers, 12 test couriers)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    courier_train_idx, courier_test_idx = next(gss.split(df, groups=df["courier_id"]))
    
    df_imputed = df.copy()
    for col, val in impute_values.items():
        df_imputed[col] = df_imputed[col].fillna(val)
        
    courier_train_df = df_imputed.iloc[courier_train_idx].copy().reset_index(drop=True)
    courier_test_df = df_imputed.iloc[courier_test_idx].copy().reset_index(drop=True)
    print(f"Courier-Grouped Split: {courier_train_df['courier_id'].nunique()} train couriers ({len(courier_train_df)} rows) vs {courier_test_df['courier_id'].nunique()} test couriers ({len(courier_test_df)} rows)")
    
    # Random 80/20 Split
    rand_train_df, rand_test_df = train_test_split(df_imputed, test_size=0.20, random_state=42, stratify=df_imputed["high_delay"])
    print(f"Random 80/20 Split: {len(rand_train_df)} train rows vs {len(rand_test_df)} test rows")
    
    print("\n" + "=" * 75)
    print("STEP 5: PERSISTING DATASETS & METADATA")
    print("=" * 75)
    
    # Save temporal splits (primary production partition)
    train_df.to_parquet(OUTPUT_DIR / "train_features.parquet", index=False)
    val_df.to_parquet(OUTPUT_DIR / "val_features.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test_features.parquet", index=False)
    
    # Save grouped & random splits
    courier_train_df.to_parquet(OUTPUT_DIR / "courier_grouped_train.parquet", index=False)
    courier_test_df.to_parquet(OUTPUT_DIR / "courier_grouped_test.parquet", index=False)
    rand_train_df.to_parquet(OUTPUT_DIR / "random_train.parquet", index=False)
    rand_test_df.to_parquet(OUTPUT_DIR / "random_test.parquet", index=False)
    
    metadata = {
        "dataset_name": "Jilin Last-Mile Express Logistics (Cleaned & Point-in-Time)",
        "total_samples": total_n,
        "temporal_split": {
            "train_samples": len(train_df),
            "train_date_range": [str(train_df["accept_time"].min()), str(train_df["accept_time"].max())],
            "val_samples": len(val_df),
            "val_date_range": [str(val_df["accept_time"].min()), str(val_df["accept_time"].max())],
            "test_samples": len(test_df),
            "test_date_range": [str(test_df["accept_time"].min()), str(test_df["accept_time"].max())],
        },
        "courier_grouped_split": {
            "train_samples": len(courier_train_df),
            "test_samples": len(courier_test_df),
            "train_couriers": int(courier_train_df["courier_id"].nunique()),
            "test_couriers": int(courier_test_df["courier_id"].nunique())
        },
        "random_split": {
            "train_samples": len(rand_train_df),
            "test_samples": len(rand_test_df)
        },
        "high_delay_threshold_minutes": high_delay_threshold,
        "impute_values": impute_values,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "meta_cols": meta_cols,
        "categorical_cols": ["time_period"],
        "numerical_cols": [col for col in feature_cols if col != "time_period"]
    }
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    with open(RESULTS_DIR / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Successfully generated all processed datasets and metadata at {OUTPUT_DIR}")

if __name__ == "__main__":
    run_preprocessing()
