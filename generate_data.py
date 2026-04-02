"""
generate_data.py
----------------
Creates a synthetic Sri Lankan house price dataset
and saves it to data/raw_houses.csv.
Prices are in Sri Lankan Rupees (LKR).
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

districts = ["Colombo", "Kandy", "Galle", "Negombo", "Jaffna", "Matara",
             "Kurunegala", "Anuradhapura", "Ratnapura", "Badulla"]

property_types = ["House", "Apartment", "Villa", "Land & House"]

district = np.random.choice(
    districts, N,
    p=[0.28, 0.15, 0.12, 0.10, 0.07, 0.07, 0.07, 0.06, 0.05, 0.03]
)
property_type = np.random.choice(
    property_types, N,
    p=[0.45, 0.25, 0.15, 0.15]
)

bedrooms       = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.05, 0.20, 0.35, 0.25, 0.10, 0.05])
bathrooms      = np.clip(bedrooms - np.random.choice([0, 1], N, p=[0.4, 0.6]), 1, None)
floors         = np.random.choice([1, 2, 3], N, p=[0.50, 0.38, 0.12])
land_perches   = np.round(np.random.exponential(scale=15, size=N) + 4, 1)
land_perches   = np.clip(land_perches, 4, 120)
floor_area_sqft = (bedrooms * 350 + np.random.normal(0, 100, N) + floors * 150).astype(int)
floor_area_sqft = np.clip(floor_area_sqft, 400, 8000)
_age_p_raw     = [0.05]*5 + [0.04]*5 + [0.03]*5 + [0.02]*10 + [0.01]*16
_age_p         = [v / sum(_age_p_raw) for v in _age_p_raw]
age_years      = np.random.choice(range(0, 41), N, p=_age_p)
has_garage     = np.random.choice([1, 0], N, p=[0.40, 0.60])
has_pool       = np.random.choice([1, 0], N, p=[0.12, 0.88])
furnished      = np.random.choice([1, 0], N, p=[0.35, 0.65])

# ── Price formula ─────────────────────────────────────────────────────────────
district_multiplier = {
    "Colombo":      3.50,
    "Negombo":      2.00,
    "Galle":        1.80,
    "Kandy":        1.60,
    "Jaffna":       1.10,
    "Matara":       1.00,
    "Kurunegala":   0.90,
    "Ratnapura":    0.85,
    "Anuradhapura": 0.80,
    "Badulla":      0.75,
}
property_type_multiplier = {
    "Villa":        1.60,
    "Land & House": 1.30,
    "House":        1.00,
    "Apartment":    0.80,
}

land_price_per_perch = {
    "Colombo":      800_000,
    "Negombo":      450_000,
    "Galle":        400_000,
    "Kandy":        350_000,
    "Jaffna":       250_000,
    "Matara":       220_000,
    "Kurunegala":   180_000,
    "Ratnapura":    160_000,
    "Anuradhapura": 150_000,
    "Badulla":      130_000,
}

base_price = np.zeros(N)
for i in range(N):
    d = district[i]
    p = property_type[i]
    land_val      = land_perches[i] * land_price_per_perch[d]
    build_val     = floor_area_sqft[i] * 18_000       # LKR 18,000 per sqft construction
    bedroom_bonus = (bedrooms[i] - 1) * 1_200_000
    garage_bonus  = has_garage[i] * 2_500_000
    pool_bonus    = has_pool[i] * 6_000_000
    furnished_bonus = furnished[i] * 1_500_000
    age_discount  = max(0, age_years[i] * 0.008)      # 0.8% value loss per year
    base_price[i] = (
        (land_val + build_val + bedroom_bonus + garage_bonus + pool_bonus + furnished_bonus)
        * property_type_multiplier[p]
        * district_multiplier[d]
        * (1 - age_discount)
    )

noise       = np.random.normal(0, 0.08, N)
price_lkr   = (base_price * (1 + noise)).astype(int)
price_lkr   = np.maximum(price_lkr, 2_500_000)  # minimum LKR 2.5M

df = pd.DataFrame({
    "district":       district,
    "property_type":  property_type,
    "bedrooms":       bedrooms,
    "bathrooms":      bathrooms,
    "floors":         floors,
    "land_perches":   land_perches,
    "floor_area_sqft": floor_area_sqft,
    "age_years":      age_years,
    "has_garage":     has_garage,
    "has_pool":       has_pool,
    "furnished":      furnished,
    "price_lkr":      price_lkr,
})

# Inject data quality issues
missing_idx = np.random.choice(N, 30, replace=False)
df.loc[missing_idx[:15], "floor_area_sqft"] = None
df.loc[missing_idx[15:], "land_perches"]    = None
duplicates = df.sample(10, random_state=1)
df = pd.concat([df, duplicates], ignore_index=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/raw_houses.csv", index=False)

print("=" * 55)
print(" Dataset generated: data/raw_houses.csv  (LKR)")
print("=" * 55)
print(f"  Total rows      : {len(df)}")
print(f"  Columns         : {list(df.columns)}")
print(f"  Min price       : LKR {df['price_lkr'].min():,}")
print(f"  Max price       : LKR {df['price_lkr'].max():,}")
print(f"  Avg price       : LKR {int(df['price_lkr'].mean()):,}")
