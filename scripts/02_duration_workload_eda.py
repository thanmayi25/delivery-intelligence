from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf




# ============================================================
# 1. LOCATE AND LOAD THE PARQUET FILE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "delivery"
)

REPORT_FOLDER = PROJECT_ROOT / "reports"

REPORT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

parquet_files = list(
    DATA_FOLDER.glob("*.parquet")
)

if not parquet_files:
    raise FileNotFoundError(
        f"No .parquet file found inside: {DATA_FOLDER}"
    )

DATA_PATH = parquet_files[0]

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

print("File:", DATA_PATH.name)

df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. CONVERT TIMESTAMP COLUMNS
# ============================================================

time_columns = [
    "accept_time",
    "accept_gps_time",
    "delivery_time",
    "delivery_gps_time",
]

for column in time_columns:

    if column in df.columns:

        df[column] = pd.to_datetime(
            df[column],
            format="%m-%d %H:%M:%S",
            errors="coerce",
        )


# ============================================================
# 3. CHECK TIMESTAMP PARSING
# ============================================================

print("\n" + "=" * 70)
print("TIMESTAMP PARSING CHECK")
print("=" * 70)

for column in time_columns:

    if column in df.columns:

        print(f"\n{column}:")

        print(
            "  Missing after parsing:",
            df[column].isna().sum()
        )

        print(
            "  Minimum:",
            df[column].min()
        )

        print(
            "  Maximum:",
            df[column].max()
        )


# ============================================================
# 4. CLEAN GPS COORDINATES
# ============================================================

gps_columns = [
    "accept_gps_lng",
    "accept_gps_lat",
    "delivery_gps_lng",
    "delivery_gps_lat",
]

# Check that all GPS columns exist

missing_gps_columns = [
    column
    for column in gps_columns
    if column not in df.columns
]

if missing_gps_columns:

    raise KeyError(
        f"Missing GPS columns: {missing_gps_columns}"
    )


# Convert GPS columns to numeric

for column in gps_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Treat near-zero GPS values as missing

for column in gps_columns:

    df.loc[
        df[column].abs() < 1,
        column
    ] = np.nan


print("\n" + "=" * 70)
print("GPS MISSING VALUES AFTER CLEANING")
print("=" * 70)

print(
    df[gps_columns].isna().sum()
)


# ============================================================
# 5. HAVERSINE DISTANCE FUNCTION
# ============================================================

def haversine_distance_km(
    longitude_1,
    latitude_1,
    longitude_2,
    latitude_2
):

    longitude_1, latitude_1, longitude_2, latitude_2 = map(
        np.radians,
        [
            longitude_1,
            latitude_1,
            longitude_2,
            latitude_2,
        ],
    )

    difference_longitude = (
        longitude_2 - longitude_1
    )

    difference_latitude = (
        latitude_2 - latitude_1
    )

    a = (
        np.sin(difference_latitude / 2) ** 2
        + np.cos(latitude_1)
        * np.cos(latitude_2)
        * np.sin(difference_longitude / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    earth_radius_km = 6371

    return earth_radius_km * c


# ============================================================
# 6. CALCULATE DELIVERY DISTANCE
# ============================================================

df["delivery_distance_km"] = haversine_distance_km(
    df["accept_gps_lng"],
    df["accept_gps_lat"],
    df["delivery_gps_lng"],
    df["delivery_gps_lat"],
)


# ============================================================
# 7. DELIVERY DURATION
# ============================================================

df["delivery_duration_minutes"] = (
    df["delivery_time"]
    - df["accept_time"]
).dt.total_seconds() / 60


# ============================================================
# 8. DELIVERY DURATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DELIVERY DURATION SUMMARY")
print("=" * 70)

print(
    df["delivery_duration_minutes"].describe()
)

print(
    "Missing duration:",
    df["delivery_duration_minutes"].isna().sum()
)

print(
    "Negative duration:",
    (
        df["delivery_duration_minutes"] < 0
    ).sum()
)


# ============================================================
# 9. DISTANCE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEANED DISTANCE SUMMARY")
print("=" * 70)

print(
    df["delivery_distance_km"].describe()
)

print(
    "Missing distance:",
    df["delivery_distance_km"].isna().sum()
)

print(
    "Zero distance:",
    (
        df["delivery_distance_km"] == 0
    ).sum()
)

print(
    "Distances above 50 km:",
    (
        df["delivery_distance_km"] > 50
    ).sum()
)

print(
    "Distances above 100 km:",
    (
        df["delivery_distance_km"] > 100
    ).sum()
)


# ============================================================
# 10. SAVE CLEANED DATASET
# ============================================================

OUTPUT_PATH = (
    REPORT_FOLDER
    / "delivery_cleaned_with_distance.parquet"
)

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("CLEANED DATASET SAVED")
print("=" * 70)

print("Saved to:", OUTPUT_PATH)
print("Final dataset shape:", df.shape)
# ============================================================
# 5. Inspect unusually large durations
# ============================================================

print("\n" + "=" * 70)
print("LARGE DELIVERY DURATIONS")
print("=" * 70)

print(
    df[
        [
            "order_id",
            "courier_id",
            "accept_time",
            "delivery_time",
            "delivery_duration_minutes",
        ]
    ]
    .sort_values("delivery_duration_minutes", ascending=False)
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 6. Orders per courier
# ============================================================

print("\n" + "=" * 70)
print("ORDERS PER COURIER")
print("=" * 70)

orders_per_courier = (
    df.groupby("courier_id")
    .size()
    .sort_values(ascending=False)
)

print("Number of couriers:", orders_per_courier.size)
print("\nSummary:")
print(orders_per_courier.describe())

print("\nTop 10 couriers:")
print(orders_per_courier.head(10))


# ============================================================
# 7. Orders per date code
# ============================================================

print("\n" + "=" * 70)
print("ORDERS PER DS VALUE")
print("=" * 70)

print("Unique ds values:", df["ds"].nunique())
print("Minimum ds:", df["ds"].min())
print("Maximum ds:", df["ds"].max())

orders_per_ds = df.groupby("ds").size()

print("\nOrders per ds summary:")
print(orders_per_ds.describe())

print("\nFirst 20 ds values:")
print(orders_per_ds.head(20))


# ============================================================
# 8. Sort records by courier and acceptance time
# ============================================================

df = df.sort_values(
    ["courier_id", "accept_time", "order_id"]
).reset_index(drop=True)


# ============================================================
# 9. Create simple recent workload features
# ============================================================

# Number of previously accepted tasks by the same courier.
df["previous_tasks"] = (
    df.groupby("courier_id")
    .cumcount()
)

# Time since the previous accepted task by the same courier.
df["previous_accept_time"] = (
    df.groupby("courier_id")["accept_time"]
    .shift(1)
)

df["minutes_since_previous_accept"] = (
    df["accept_time"] - df["previous_accept_time"]
).dt.total_seconds() / 60


# ------------------------------------------------------------
# Calculate recent workload separately for each courier
# ------------------------------------------------------------

def add_workload_features(group):

    group = group.sort_values(
        ["accept_time", "order_id"]
    ).copy()

    time_indexed = group.set_index("accept_time")

    group["tasks_previous_1h"] = (
        time_indexed["order_id"]
        .rolling("1h", closed="left")
        .count()
        .to_numpy()
    )

    group["tasks_previous_3h"] = (
        time_indexed["order_id"]
        .rolling("3h", closed="left")
        .count()
        .to_numpy()
    )

    return group


df = (
    df.groupby("courier_id", group_keys=False)
    .apply(add_workload_features)
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Replace missing previous-task counts with zero
# ------------------------------------------------------------

df["tasks_previous_1h"] = (
    df["tasks_previous_1h"]
    .fillna(0)
    .astype(int)
)

df["tasks_previous_3h"] = (
    df["tasks_previous_3h"]
    .fillna(0)
    .astype(int)
)
# ============================================================
# 10. Create time features
# ============================================================

df["accept_hour"] = df["accept_time"].dt.hour

df["accept_minute"] = df["accept_time"].dt.minute

df["accept_day"] = df["accept_time"].dt.day

df["accept_weekday"] = (
    df["accept_time"].dt.dayofweek
)

df["is_weekend"] = (
    df["accept_weekday"] >= 5
).astype(int)


def assign_time_period(hour):

    if 6 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 17:
        return "Afternoon"

    elif 17 <= hour < 22:
        return "Evening"

    else:
        return "Night"


df["time_period"] = (
    df["accept_hour"].apply(assign_time_period)
)
# ============================================================
# 11. Inspect workload features
# ============================================================

print("\n" + "=" * 70)
print("WORKLOAD FEATURES")
print("=" * 70)

workload_columns = [
    "previous_tasks",
    "minutes_since_previous_accept",
    "tasks_previous_1h",
    "tasks_previous_3h",
]

print(df[workload_columns].describe())

print("\nSample workload records:")
print(
    df[
        [
            "courier_id",
            "accept_time",
            "delivery_duration_minutes",
            "previous_tasks",
            "minutes_since_previous_accept",
            "tasks_previous_1h",
            "tasks_previous_3h",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

# ============================================================
# 11. Time period vs performance
# ============================================================

time_workload_summary = (
    df.groupby("time_period")
    .agg(
        orders=("order_id", "count"),

        average_duration_minutes=(
            "delivery_duration_minutes",
            "mean",
        ),

        median_duration_minutes=(
            "delivery_duration_minutes",
            "median",
        ),

        average_workload_1h=(
            "tasks_previous_1h",
            "mean",
        ),

        average_workload_3h=(
            "tasks_previous_3h",
            "mean",
        ),

        average_distance_km=(
            "delivery_distance_km",
            "mean",
        ),
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("TIME PERIOD VS PERFORMANCE")
print("=" * 70)

print(
    time_workload_summary.to_string(index=False)
)
# ============================================================
# 12. Workload, distance and time analysis
# ============================================================

df["distance_group"] = pd.cut(
    df["delivery_distance_km"],
    bins=[
        -1,
        1,
        3,
        5,
        10,
        float("inf"),
    ],
    labels=[
        "0–1 km",
        "1–3 km",
        "3–5 km",
        "5–10 km",
        "10+ km",
    ],
)

workload_distance_summary = (
    df.groupby(
        ["distance_group", "time_period"],
        observed=False,
    )
    .agg(
        orders=("order_id", "count"),

        average_duration_minutes=(
            "delivery_duration_minutes",
            "mean",
        ),

        median_duration_minutes=(
            "delivery_duration_minutes",
            "median",
        ),

        average_workload_1h=(
            "tasks_previous_1h",
            "mean",
        ),
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("WORKLOAD, DISTANCE AND TIME ANALYSIS")
print("=" * 70)

print(
    workload_distance_summary.to_string(index=False)

)
# ============================================================
# 13. Fixed workload group analysis
# ============================================================

analysis_df = df[
    [
        "delivery_duration_minutes",
        "tasks_previous_1h",
        "tasks_previous_3h",
        "delivery_distance_km",
        "time_period",
        "courier_id",
    ]
].dropna()

analysis_df = analysis_df[
    analysis_df["delivery_duration_minutes"] >= 0
]

analysis_df["workload_1h_group"] = pd.cut(
    analysis_df["tasks_previous_1h"],
    bins=[
        -1,
        0,
        1,
        3,
        5,
        10,
        20,
        float("inf"),
    ],
    labels=[
        "0 tasks",
        "1 task",
        "2–3 tasks",
        "4–5 tasks",
        "6–10 tasks",
        "11–20 tasks",
        "21+ tasks",
    ],
)

fixed_workload_summary = (
    analysis_df
    .groupby(
        "workload_1h_group",
        observed=False,
    )
    .agg(
        orders=("delivery_duration_minutes", "size"),

        average_duration_minutes=(
            "delivery_duration_minutes",
            "mean",
        ),

        median_duration_minutes=(
            "delivery_duration_minutes",
            "median",
        ),

        p90_duration_minutes=(
            "delivery_duration_minutes",
            lambda x: x.quantile(0.90),
        ),
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("FIXED WORKLOAD GROUP SUMMARY")
print("=" * 70)

print(
    fixed_workload_summary.to_string(index=False)
)


# ============================================================
# 14. Correlations — descriptive only
# ============================================================

correlation_columns = [
    "delivery_duration_minutes",
    "tasks_previous_1h",
    "tasks_previous_3h",
    "delivery_distance_km",
]

print("\n" + "=" * 70)
print("CORRELATIONS")
print("=" * 70)

print(
    analysis_df[correlation_columns]
    .corr(numeric_only=True)
    .round(3)
)

# ============================================================
# 13. Save processed preliminary EDA data
# ============================================================

output_columns = [
    "order_id",
    "courier_id",
    "region_id",
    "city",

    "lng",
    "lat",

    "accept_time",
    "delivery_time",

    "delivery_duration_minutes",
    "delivery_distance_km",

    "previous_tasks",
    "previous_accept_time",
    "minutes_since_previous_accept",

    "tasks_previous_1h",
    "tasks_previous_3h",

    "accept_hour",
    "accept_minute",
    "accept_day",
    "accept_weekday",
    "is_weekend",
    "time_period",
    "distance_group",

    "ds",
]

available_output_columns = [
    column for column in output_columns if column in df.columns
]

df[available_output_columns].to_csv(
    REPORT_FOLDER / "preliminary_duration_workload_data.csv",
    index=False,
)

fixed_workload_summary.to_csv(
    REPORT_FOLDER / "workload_duration_summary.csv",
    index=False,
)


# ============================================================
# 14. Create duration plot
# ============================================================

plot_df = df[
    "delivery_duration_minutes"
].dropna()

plot_df = plot_df[plot_df >= 0]

# Restrict only the plot to the 99th percentile
# so extreme values do not hide the main distribution.
upper_limit = plot_df.quantile(0.99)

plot_df = plot_df[plot_df <= upper_limit]

plt.figure(figsize=(10, 6))
plt.hist(plot_df, bins=50)

plt.xlabel("Delivery duration in minutes")
plt.ylabel("Number of orders")
plt.title("Delivery Duration Distribution — Jilin")

plt.tight_layout()

plt.savefig(
    REPORT_FOLDER / "delivery_duration_distribution.png",
    dpi=150,
)

plt.close()


# ============================================================
# 15. Create workload comparison plot
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    fixed_workload_summary["workload_1h_group"].astype(str),
    fixed_workload_summary["average_duration_minutes"],
)
plt.xlabel("Tasks accepted in previous 1 hour")
plt.ylabel("Average delivery duration in minutes")
plt.title("Workload Group vs Average Delivery Duration")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    REPORT_FOLDER / "workload_vs_duration.png",
    dpi=150,
)

plt.close()


print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print(
    "Saved:",
    REPORT_FOLDER / "preliminary_duration_workload_data.csv",
)

print(
    "Saved:",
    REPORT_FOLDER / "workload_duration_summary.csv",
)

print(
    "Saved:",
    REPORT_FOLDER / "delivery_duration_distribution.png",
)

print(
    "Saved:",
    REPORT_FOLDER / "workload_vs_duration.png",
)
print("\nLARGEST DISTANCES")

print(
    df[
        [
            "order_id",
            "accept_gps_lng",
            "accept_gps_lat",
            "delivery_gps_lng",
            "delivery_gps_lat",
            "delivery_distance_km",
        ]
    ]
    .sort_values(
        "delivery_distance_km",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)
print(df[
    [
        "accept_gps_lng",
        "accept_gps_lat",
        "delivery_gps_lng",
        "delivery_gps_lat",
    ]
].describe())

# ============================================================
# GPS DISTANCE OUTLIER INVESTIGATION
# ============================================================

distance_columns = [
    "order_id",
    "accept_gps_lng",
    "accept_gps_lat",
    "delivery_gps_lng",
    "delivery_gps_lat",
    "delivery_distance_km",
]

# Top 20 largest distances

largest_distance_orders = (
    df.nlargest(
        20,
        "delivery_distance_km"
    )[distance_columns]
)

print("\n" + "=" * 70)
print("TOP 20 LARGEST DELIVERY DISTANCES")
print("=" * 70)

print(
    largest_distance_orders.to_string(
        index=False
    )
)


# ============================================================
# FLAG LONG-DISTANCE ORDERS
# ============================================================

df["long_distance_flag"] = (
    df["delivery_distance_km"] > 50
)

print("\n" + "=" * 70)
print("LONG-DISTANCE ORDER COUNTS")
print("=" * 70)

print(
    df["long_distance_flag"].value_counts()
)


# ============================================================
# INVESTIGATE LONG-DISTANCE ORDERS
# ============================================================

long_distance_columns = [
    "order_id",
    "accept_time",
    "delivery_time",
    "delivery_duration_minutes",
    "accept_gps_lng",
    "accept_gps_lat",
    "delivery_gps_lng",
    "delivery_gps_lat",
    "delivery_distance_km",
]

# Filter orders with distance > 50 km
long_distance_orders = (
    df[df["delivery_distance_km"] > 50][long_distance_columns]
    .sort_values("delivery_distance_km", ascending=False)
)

print("\n" + "=" * 70)
print("LONG-DISTANCE ORDER INVESTIGATION")
print("=" * 70)

# Display full table without index
print(long_distance_orders.to_string(index=False))

df = df.sort_values(
    ["courier_id", "accept_time", "order_id"]
).reset_index(drop=True)

# ============================================================
# TASK SEQUENCE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("TASK SEQUENCE ANALYSIS")
print("=" * 60)

# Sort orders in the actual sequence handled by each courier
df = df.sort_values(
    ["courier_id", "accept_time", "order_id"]
).reset_index(drop=True)

# Previous task handled by the same courier
df["previous_order_id"] = (
    df.groupby("courier_id")["order_id"].shift(1)
)

df["previous_accept_time"] = (
    df.groupby("courier_id")["accept_time"].shift(1)
)

# Time gap between consecutive accepted tasks
df["task_gap_minutes"] = (
    df["accept_time"] - df["previous_accept_time"]
).dt.total_seconds() / 60

# Previous task's delivery duration
df["previous_delivery_duration"] = (
    df.groupby("courier_id")["delivery_duration_minutes"].shift(1)
)

# Previous task acceptance GPS
df["previous_accept_gps_lng"] = (
    df.groupby("courier_id")["accept_gps_lng"].shift(1)
)

df["previous_accept_gps_lat"] = (
    df.groupby("courier_id")["accept_gps_lat"].shift(1)
)

# Haversine distance between consecutive task acceptance locations
df["previous_to_current_distance_km"] = haversine_distance_km(
    df["previous_accept_gps_lat"],
    df["previous_accept_gps_lng"],
    df["accept_gps_lat"],
    df["accept_gps_lng"]
)

# Remove first task of every courier because it has no previous task
sequence_df = df[
    df["previous_order_id"].notna()
].copy()

# Remove invalid sequence gaps
sequence_df = sequence_df[
    sequence_df["task_gap_minutes"] >= 0
].copy()

print("\nSequence records:", len(sequence_df))

print("\nSample sequence:")
print(
    sequence_df[
        [
            "courier_id",
            "previous_order_id",
            "order_id",
            "task_gap_minutes",
            "previous_to_current_distance_km",
            "previous_delivery_duration",
            "delivery_duration_minutes"
        ]
    ].head(10)
)
# ============================================================
# STEP 2: SEQUENCE PATTERN ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SEQUENCE PATTERN ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 1. Group by time gap between consecutive tasks
# ------------------------------------------------------------

sequence_df["task_gap_group"] = pd.cut(
    sequence_df["task_gap_minutes"],
    bins=[-1, 15, 30, 60, 120, float("inf")],
    labels=[
        "0-15 min",
        "15-30 min",
        "30-60 min",
        "60-120 min",
        "120+ min"
    ]
)

gap_analysis = (
    sequence_df
    .groupby("task_gap_group", observed=False)
    .agg(
        orders=("order_id", "count"),
        average_duration_minutes=(
            "delivery_duration_minutes", "mean"
        ),
        median_duration_minutes=(
            "delivery_duration_minutes", "median"
        ),
        p90_duration_minutes=(
            "delivery_duration_minutes",
            lambda x: x.quantile(0.90)
        )
    )
    .reset_index()
)

print("\n--- TIME GAP ANALYSIS ---")
print(gap_analysis.to_string(index=False))


# ------------------------------------------------------------
# 2. Group by distance between consecutive task locations
# ------------------------------------------------------------

sequence_df["sequence_distance_group"] = pd.cut(
    sequence_df["previous_to_current_distance_km"],
    bins=[-0.001, 0.1, 0.5, 1, 2, float("inf")],
    labels=[
        "0-0.1 km",
        "0.1-0.5 km",
        "0.5-1 km",
        "1-2 km",
        "2+ km"
    ]
)

distance_sequence_analysis = (
    sequence_df
    .groupby("sequence_distance_group", observed=False)
    .agg(
        orders=("order_id", "count"),
        average_duration_minutes=(
            "delivery_duration_minutes", "mean"
        ),
        median_duration_minutes=(
            "delivery_duration_minutes", "median"
        ),
        p90_duration_minutes=(
            "delivery_duration_minutes",
            lambda x: x.quantile(0.90)
        )
    )
    .reset_index()
)

print("\n--- CONSECUTIVE TASK DISTANCE ANALYSIS ---")
print(distance_sequence_analysis.to_string(index=False))


# ------------------------------------------------------------
# 3. Group by previous task duration
# ------------------------------------------------------------

sequence_df["previous_duration_group"] = pd.cut(
    sequence_df["previous_delivery_duration"],
    bins=[-1, 60, 120, 180, 240, 360, float("inf")],
    labels=[
        "0-60 min",
        "60-120 min",
        "120-180 min",
        "180-240 min",
        "240-360 min",
        "360+ min"
    ]
)

previous_duration_analysis = (
    sequence_df
    .groupby("previous_duration_group", observed=False)
    .agg(
        orders=("order_id", "count"),
        average_duration_minutes=(
            "delivery_duration_minutes", "mean"
        ),
        median_duration_minutes=(
            "delivery_duration_minutes", "median"
        ),
        p90_duration_minutes=(
            "delivery_duration_minutes",
            lambda x: x.quantile(0.90)
        )
    )
    .reset_index()
)

print("\n--- PREVIOUS TASK DURATION ANALYSIS ---")
print(previous_duration_analysis.to_string(index=False))


# ------------------------------------------------------------
# 4. Correlations between sequence variables and performance
# ------------------------------------------------------------

sequence_correlations = sequence_df[
    [
        "task_gap_minutes",
        "previous_to_current_distance_km",
        "previous_delivery_duration",
        "delivery_duration_minutes"
    ]
].corr()

print("\n--- SEQUENCE CORRELATIONS ---")
print(sequence_correlations.round(3))
# ============================================================
# STEP 3: ADJUSTED SEQUENCE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ADJUSTED SEQUENCE ANALYSIS")
print("=" * 60)

# Keep only variables needed for analysis
adjusted_df = sequence_df[
    [
        "delivery_duration_minutes",
        "previous_delivery_duration",
        "delivery_distance_km",
        "time_period"
    ]
].dropna().copy()

# Correlation after accounting for the main variables
print("\n--- BASIC CORRELATIONS ---")

print(
    adjusted_df[
        [
            "previous_delivery_duration",
            "delivery_distance_km",
            "delivery_duration_minutes"
        ]
    ].corr().round(3)
)


# ------------------------------------------------------------
# Compare current delivery duration by previous task duration
# while separating by current delivery distance
# ------------------------------------------------------------

adjusted_df["current_distance_group"] = pd.cut(
    adjusted_df["delivery_distance_km"],
    bins=[-0.001, 1, 3, 5, 10, float("inf")],
    labels=[
        "0-1 km",
        "1-3 km",
        "3-5 km",
        "5-10 km",
        "10+ km"
    ]
)

adjusted_distance_analysis = (
    adjusted_df
    .groupby(
        [
            "current_distance_group",
            pd.cut(
                adjusted_df["previous_delivery_duration"],
                bins=[-1, 120, 240, 360, float("inf")],
                labels=[
                    "0-120 min",
                    "120-240 min",
                    "240-360 min",
                    "360+ min"
                ]
            )
        ],
        observed=False
    )
    .agg(
        orders=("delivery_duration_minutes", "count"),
        average_current_duration=(
            "delivery_duration_minutes", "mean"
        ),
        median_current_duration=(
            "delivery_duration_minutes", "median"
        )
    )
    .reset_index()
)

print("\n--- PREVIOUS TASK * CURRENT DISTANCE ---")
print(adjusted_distance_analysis.to_string(index=False))


# ------------------------------------------------------------
# Previous task duration by time period
# ------------------------------------------------------------

adjusted_time_analysis = (
    adjusted_df
    .groupby(
        [
            "time_period",
            pd.cut(
                adjusted_df["previous_delivery_duration"],
                bins=[-1, 120, 240, 360, float("inf")],
                labels=[
                    "0-120 min",
                    "120-240 min",
                    "240-360 min",
                    "360+ min"
                ]
            )
        ],
        observed=False
    )
    .agg(
        orders=("delivery_duration_minutes", "count"),
        average_current_duration=(
            "delivery_duration_minutes", "mean"
        ),
        median_current_duration=(
            "delivery_duration_minutes", "median"
        )
    )
    .reset_index()
)
print("\n--- PREVIOUS TASK * TIME PERIOD ---")
print(adjusted_time_analysis.to_string(index=False))
# ============================================================
# STEP 4: WORKLOAD IMPACT TEST
# ============================================================

import statsmodels.formula.api as smf

print("\n" + "=" * 60)
print("WORKLOAD IMPACT TEST")
print("=" * 60)

# Prepare analysis data
workload_test_df = df[
    [
        "delivery_duration_minutes",
        "tasks_previous_1h",
        "tasks_previous_3h",
        "delivery_distance_km",
        "time_period"
    ]
].dropna().copy()

# Remove invalid durations
workload_test_df = workload_test_df[
    workload_test_df["delivery_duration_minutes"] >= 0
].copy()


# ------------------------------------------------------------
# MODEL 1: 1-hour workload
# ------------------------------------------------------------

model_1h = smf.ols(
    """
    delivery_duration_minutes
    ~ tasks_previous_1h
    + delivery_distance_km
    + C(time_period)
    """,
    data=workload_test_df
).fit(cov_type="HC3")

print("\n--- MODEL 1: 1-HOUR WORKLOAD ---")
print(model_1h.summary())


# ------------------------------------------------------------
# MODEL 2: 3-hour workload
# ------------------------------------------------------------

model_3h = smf.ols(
    """
    delivery_duration_minutes
    ~ tasks_previous_3h
    + delivery_distance_km
    + C(time_period)
    """,
    data=workload_test_df
).fit(cov_type="HC3")

print("\n--- MODEL 2: 3-HOUR WORKLOAD ---")
print(model_3h.summary())


# ------------------------------------------------------------
# Extract workload results
# ------------------------------------------------------------

workload_results = pd.DataFrame({
    "workload_window": ["1 hour", "3 hours"],
    "coefficient": [
        model_1h.params["tasks_previous_1h"],
        model_3h.params["tasks_previous_3h"]
    ],
    "p_value": [
        model_1h.pvalues["tasks_previous_1h"],
        model_3h.pvalues["tasks_previous_3h"]
    ],
    "r_squared": [
        model_1h.rsquared,
        model_3h.rsquared
    ]
})

print("\n--- WORKLOAD EFFECT SUMMARY ---")
print(workload_results.round(4).to_string(index=False))

print("\n" + "=" * 60)
print("WORKLOAD THRESHOLD ANALYSIS")
print("=" * 60)

# Define high-delay deliveries using the 90th percentile
delay_threshold = workload_test_df["delivery_duration_minutes"].quantile(0.90)

workload_test_df["high_delay"] = (
    workload_test_df["delivery_duration_minutes"] > delay_threshold
).astype(int)

print(f"\n90th percentile delay threshold: {delay_threshold:.2f} minutes")

print("\nHigh-delay distribution:")
print(workload_test_df["high_delay"].value_counts())
# Create workload groups
workload_test_df["workload_group"] = pd.cut(
    workload_test_df["tasks_previous_1h"],
    bins=[-1, 0, 1, 3, 5, 10, 20, float("inf")],
    labels=[
        "0 tasks",
        "1 task",
        "2–3 tasks",
        "4–5 tasks",
        "6–10 tasks",
        "11–20 tasks",
        "21+ tasks"
    ]
)

# Calculate high-delay rate for each workload group
threshold_analysis = (
    workload_test_df
    .groupby("workload_group", observed=False)
    .agg(
        orders=("high_delay", "size"),
        high_delay_orders=("high_delay", "sum"),
        high_delay_rate=("high_delay", "mean")
    )
    .reset_index()
)

threshold_analysis["high_delay_rate_percent"] = (
    threshold_analysis["high_delay_rate"] * 100
)

print("\nHIGH-DELAY RATE BY WORKLOAD")
print(threshold_analysis.to_string(index=False))
import statsmodels.formula.api as smf

threshold_model = smf.logit(
    """
    high_delay ~ C(workload_group)
    + delivery_distance_km
    + C(time_period)
    """,
    data=workload_test_df
).fit(disp=False)

print("\nWORKLOAD THRESHOLD LOGISTIC REGRESSION")
print(threshold_model.summary())
# Create a simple extreme-workload indicator
workload_test_df["extreme_workload"] = (
    workload_test_df["tasks_previous_1h"] >= 21
).astype(int)

print("\nExtreme workload distribution:")
print(
    workload_test_df["extreme_workload"]
    .value_counts()
    .sort_index()
)

# Logistic regression with extreme workload
extreme_model = smf.logit(
    """
    high_delay ~ extreme_workload
    + delivery_distance_km
    + C(time_period)
    """,
    data=workload_test_df
).fit_regularized(
    method="l1",
    alpha=0.01,
    disp=False
)

print("\nEXTREME WORKLOAD LOGISTIC REGRESSION")
print(extreme_model.summary())