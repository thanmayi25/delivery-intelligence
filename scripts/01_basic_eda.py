from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Locate the downloaded Parquet file
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data" / "raw" / "delivery"

parquet_files = list(DATA_FOLDER.glob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"No Parquet file found inside: {DATA_FOLDER}"
    )

if len(parquet_files) > 1:
    print("Multiple Parquet files found:")
    for file in parquet_files:
        print(" -", file.name)

DATA_PATH = parquet_files[0]

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)
print("File:", DATA_PATH.name)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_parquet(DATA_PATH)


# --------------------------------------------------
# 3. Basic dataset information
# --------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(df.dtypes)


print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
print(df.head().to_string())


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_values = df.isna().sum()
missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    print("No missing values found.")
else:
    print(missing_values.sort_values(ascending=False))


print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)
print("Duplicate rows:", df.duplicated().sum())


print("\n" + "=" * 70)
print("UNIQUE VALUES FOR POSSIBLE ID COLUMNS")
print("=" * 70)

possible_id_columns = [
    "order_id",
    "courier_id",
    "region_id",
    "city",
    "task_id",
    "delivery_id"
]

for column in possible_id_columns:
    if column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")


print("\n" + "=" * 70)
print("NUMERIC SUMMARY")
print("=" * 70)
print(df.describe().T)


print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)