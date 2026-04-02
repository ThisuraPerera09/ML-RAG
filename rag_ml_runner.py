"""
rag_ml_runner.py — House Price ML Pipeline
------------------------------------------
Training phase:  train_and_save(df)            — df WITH price_lkr
Prediction phase: predict_with_saved_model(df) — df WITHOUT price_lkr
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR     = "model"
MODEL_PATH    = os.path.join(MODEL_DIR, "model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH)


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = ["land_perches", "floor_area_sqft", "age_years", "bedrooms",
                    "bathrooms", "floors"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates()

    if "land_perches"    in df.columns: df = df[df["land_perches"]    > 0]
    if "floor_area_sqft" in df.columns: df = df[df["floor_area_sqft"] > 0]
    if "bedrooms"        in df.columns: df = df[df["bedrooms"].between(1, 20)]

    for col in ["bedrooms", "bathrooms", "floors", "age_years",
                "has_garage", "has_pool", "furnished"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df.reset_index(drop=True)


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived features
    if "floor_area_sqft" in df.columns and "bedrooms" in df.columns:
        df["sqft_per_bedroom"] = (df["floor_area_sqft"] / df["bedrooms"]).round(1)
    if "land_perches" in df.columns and "floor_area_sqft" in df.columns:
        df["build_to_land_ratio"] = (df["floor_area_sqft"] / (df["land_perches"] * 272.25)).round(3)
    if "age_years" in df.columns:
        df["is_new"]    = (df["age_years"] <= 5).astype(int)
        df["age_decade"] = (df["age_years"] // 10).astype(int)
    if "bedrooms" in df.columns:
        df["is_large_home"] = (df["bedrooms"] >= 4).astype(int)
    if "has_pool" in df.columns and "has_garage" in df.columns:
        df["premium_features"] = df["has_pool"] + df["has_garage"]

    # Ordinal: property type by typical price tier
    prop_order = {"Apartment": 0, "House": 1, "Land & House": 2, "Villa": 3}
    if "property_type" in df.columns:
        df["property_type_score"] = df["property_type"].map(prop_order).fillna(1)

    # One-hot: district
    if "district" in df.columns:
        dummies = pd.get_dummies(df["district"], prefix="district", drop_first=True, dtype=int)
        df = pd.concat([df, dummies], axis=1)

    # Drop text columns
    text_cols = ["district", "property_type"]
    df = df.drop(columns=[c for c in text_cols if c in df.columns])

    return df


# ── Analytical stats ──────────────────────────────────────────────────────────

def compute_analytical_stats(pred_df: pd.DataFrame, feature_importance: dict) -> dict:
    cost_col = "estimated_price"

    def group_stats(df, col):
        if col not in df.columns:
            return {}
        return (
            df.groupby(col)[cost_col]
            .agg(avg="mean", count="count")
            .sort_values("avg", ascending=False)
            .apply(lambda r: {"avg": int(r["avg"]), "count": int(r["count"])}, axis=1)
            .to_dict()
        )

    pred_df = pred_df.copy()

    # Bedroom buckets
    pred_df["bedroom_bucket"] = pd.cut(
        pred_df["bedrooms"], bins=[0, 2, 3, 4, 20],
        labels=["1–2 beds", "3 beds", "4 beds", "5+ beds"]
    )
    bedroom_stats = (
        pred_df.groupby("bedroom_bucket", observed=True)[cost_col]
        .mean().round(0).astype(int).to_dict()
    )

    # Land size buckets
    pred_df["land_bucket"] = pd.cut(
        pred_df["land_perches"], bins=[0, 8, 15, 30, 200],
        labels=["< 8 perches", "8–15 perches", "15–30 perches", "30+ perches"]
    )
    land_stats = (
        pred_df.groupby("land_bucket", observed=True)[cost_col]
        .mean().round(0).astype(int).to_dict()
    )

    # Age buckets
    pred_df["age_bucket"] = pd.cut(
        pred_df["age_years"], bins=[-1, 5, 10, 20, 100],
        labels=["New (0–5 yrs)", "5–10 yrs", "10–20 yrs", "20+ yrs"]
    )
    age_stats = (
        pred_df.groupby("age_bucket", observed=True)[cost_col]
        .mean().round(0).astype(int).to_dict()
    )

    # Group feature importance
    dist_importance  = sum(v for k, v in feature_importance.items() if k.startswith("district_"))
    other_importance = {k: v for k, v in feature_importance.items() if not k.startswith("district_")}
    grouped_importance = dict(sorted(
        {**other_importance, "district (all)": round(dist_importance, 4)}.items(),
        key=lambda x: x[1], reverse=True
    ))

    return {
        "district_stats":     group_stats(pred_df, "district"),
        "property_type_stats": group_stats(pred_df, "property_type"),
        "bedroom_stats":      bedroom_stats,
        "land_stats":         land_stats,
        "age_stats":          age_stats,
        "grouped_importance": grouped_importance,
        "overall_avg":        int(pred_df[cost_col].mean()),
    }


# ── Training phase ────────────────────────────────────────────────────────────

def train_and_save(df: pd.DataFrame) -> dict:
    df_clean = clean_data(df)
    df_feat  = engineer_features(df_clean.copy())

    TARGET   = "price_lkr"
    FEATURES = [c for c in df_feat.columns if c != TARGET]

    X = df_feat[FEATURES]
    y = df_feat[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2   = r2_score(y_test, y_pred_test)
    mape = (abs(y_test.values - y_pred_test) / y_test.values).mean() * 100

    metrics = {
        "mae":        round(mae, 2),
        "rmse":       round(rmse, 2),
        "r2":         round(r2, 4),
        "accuracy":   round(100 - mape, 2),
        "mape":       round(mape, 2),
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }

    feature_importance = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURES, model.feature_importances_)
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURES, f)

    # Predict on full training set so RAG can be built from all 5000 houses
    y_pred_all = np.maximum(model.predict(X).round(0).astype(int), 0)
    pred_df    = df_clean.copy()
    pred_df["estimated_price"] = y_pred_all

    analytical_stats = compute_analytical_stats(pred_df, feature_importance)

    dist_col      = "district"      if "district"      in pred_df.columns else None
    prop_type_col = "property_type" if "property_type" in pred_df.columns else None

    dataset_stats = {
        "total_properties":        len(pred_df),
        "avg_estimated_price":     int(pred_df["estimated_price"].mean()),
        "min_estimated_price":     int(pred_df["estimated_price"].min()),
        "max_estimated_price":     int(pred_df["estimated_price"].max()),
        "avg_bedrooms":            int(pred_df["bedrooms"].mean()) if "bedrooms" in pred_df.columns else 0,
        "districts":               pred_df[dist_col].unique().tolist() if dist_col else [],
        "district_avg_prices": (
            pred_df.groupby(dist_col)["estimated_price"]
            .mean().round(0).astype(int).to_dict()
        ) if dist_col else {},
        "property_type_avg_prices": (
            pred_df.groupby(prop_type_col)["estimated_price"]
            .mean().round(0).astype(int).to_dict()
        ) if prop_type_col else {},
    }

    return {
        "metrics":            metrics,
        "feature_importance": feature_importance,
        "feature_count":      len(FEATURES),
        "train_rows":         len(df_clean),
        # Full pipeline_results shape for load_and_embed
        "pipeline_results": {
            "predictions_df":   pred_df,
            "metrics":          metrics,
            "feature_importance": feature_importance,
            "dataset_stats":    dataset_stats,
            "features":         FEATURES,
            "analytical_stats": analytical_stats,
        },
    }


# ── Prediction phase ──────────────────────────────────────────────────────────

def predict_with_saved_model(df: pd.DataFrame) -> dict:
    if not model_exists():
        raise FileNotFoundError("No trained model found. Upload training data first.")

    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        train_features = json.load(f)

    df_clean    = clean_data(df)
    df_for_feat = df_clean.drop(columns=["price_lkr"], errors="ignore")
    df_feat     = engineer_features(df_for_feat.copy())

    X      = df_feat.reindex(columns=train_features, fill_value=0)
    y_pred = np.maximum(model.predict(X).round(0).astype(int), 0)

    pred_df = df_clean.copy()
    pred_df["estimated_price"] = y_pred

    feature_importance = {
        feat: round(float(imp), 4)
        for feat, imp in zip(train_features, model.feature_importances_)
    }

    analytical_stats = compute_analytical_stats(pred_df, feature_importance)

    dist_col      = "district"       if "district"      in pred_df.columns else None
    prop_type_col = "property_type"  if "property_type" in pred_df.columns else None

    dataset_stats = {
        "total_properties":   len(pred_df),
        "avg_estimated_price": int(pred_df["estimated_price"].mean()),
        "min_estimated_price": int(pred_df["estimated_price"].min()),
        "max_estimated_price": int(pred_df["estimated_price"].max()),
        "avg_bedrooms":        int(pred_df["bedrooms"].mean()) if "bedrooms" in pred_df.columns else 0,
        "districts":           pred_df[dist_col].unique().tolist() if dist_col else [],
        "district_avg_prices": (
            pred_df.groupby(dist_col)["estimated_price"]
            .mean().round(0).astype(int).to_dict()
        ) if dist_col else {},
        "property_type_avg_prices": (
            pred_df.groupby(prop_type_col)["estimated_price"]
            .mean().round(0).astype(int).to_dict()
        ) if prop_type_col else {},
    }

    return {
        "predictions_df":   pred_df,
        "metrics":          {},
        "feature_importance": feature_importance,
        "dataset_stats":    dataset_stats,
        "features":         train_features,
        "analytical_stats": analytical_stats,
    }
