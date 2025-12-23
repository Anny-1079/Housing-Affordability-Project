import pandas as pd
import numpy as np

np.random.seed(42)

# --------------------------------------------------
# LOAD ORIGINAL DATASET (KEEP EVERYTHING)
# --------------------------------------------------
df = pd.read_csv("data/Bengaluru_House_Data.csv")
print("Original shape:", df.shape)

# --------------------------------------------------
# IDENTIFY PRICE COLUMN
# --------------------------------------------------
price_col = None
for col in df.columns:
    if "price" in col.lower():
        price_col = col
        break

if price_col is None:
    raise ValueError("❌ No price column found in dataset")

# Drop rows without price
df = df.dropna(subset=[price_col])

# Convert price (Lakhs → INR)
df["price_inr"] = df[price_col].astype(float) * 100_000

# --------------------------------------------------
# NORMALIZED PRICE (for realistic conditioning)
# --------------------------------------------------
price_norm = df["price_inr"] / df["price_inr"].max()

# --------------------------------------------------
# ADD INFRASTRUCTURE FEATURES (ROW-WISE)
# --------------------------------------------------

# ✈ Airport distance (luxury areas closer on average)
df["airport_distance_km"] = np.clip(
    np.random.normal(
        loc=22 - price_norm * 8,
        scale=5,
        size=len(df)
    ),
    6, 45
).round(2)

# 🚆 Railway station distance (FIXED: per-row randomness)
df["railway_station_distance_km"] = np.clip(
    np.random.normal(
        loc=5.5 - price_norm * 2,
        scale=2.2,
        size=len(df)
    ),
    0.5, 15
).round(2)

# 🏥 Number of hospitals nearby (count, not distance)
df["hospitals_count"] = np.clip(
    (price_norm * 6).astype(int) +
    np.random.randint(1, 3, size=len(df)),
    1, 10
)

# 🏥 Nearest hospital distance
df["nearest_hospital_distance_km"] = np.clip(
    np.random.normal(
        loc=3.5 - price_norm * 1.2,
        scale=1.5,
        size=len(df)
    ),
    0.3, 12
).round(2)

# 🏫 Number of schools nearby
df["schools_count"] = np.clip(
    (price_norm * 8).astype(int) +
    np.random.randint(1, 4, size=len(df)),
    1, 15
)

# --------------------------------------------------
# SAVE AUGMENTED DATASET (ALL ORIGINAL + NEW)
# --------------------------------------------------
output_path = "data/bengaluru_housing_augmented_full.csv"
df.to_csv(output_path, index=False)

print("✅ Augmented dataset saved successfully")
print("Output path:", output_path)
print("Final shape:", df.shape)
print("New columns added:")
print([
    "airport_distance_km",
    "railway_station_distance_km",
    "hospitals_count",
    "nearest_hospital_distance_km",
    "schools_count"
])
