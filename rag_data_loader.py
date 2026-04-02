"""
rag_data_loader.py — House Price ML results → analytical text documents → embeddings
"""

from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM   = 384

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


FEATURE_LABELS = {
    "land_perches":         "land size (perches)",
    "floor_area_sqft":      "floor area (sqft)",
    "bedrooms":             "number of bedrooms",
    "bathrooms":            "number of bathrooms",
    "floors":               "number of floors",
    "age_years":            "age of the property",
    "has_garage":           "garage availability",
    "has_pool":             "swimming pool",
    "furnished":            "furnished status",
    "sqft_per_bedroom":     "sqft per bedroom",
    "build_to_land_ratio":  "build-to-land ratio",
    "is_new":               "new property flag (<=5 yrs)",
    "age_decade":           "age decade",
    "is_large_home":        "large home flag (4+ beds)",
    "premium_features":     "premium features count (pool + garage)",
    "property_type_score":  "property type tier",
    "district (all)":       "district/location (combined)",
}

def _feat_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("district_"):
        return f"district: {name[9:]}"
    return name


# ── Document builders ──────────────────────────────────────────────────────────

def _district_analysis_doc(stats: dict, src: str) -> dict | None:
    district_stats = stats.get("district_stats", {})
    overall_avg    = stats.get("overall_avg", 0)
    if not district_stats:
        return None

    ranked        = sorted(district_stats.items(), key=lambda x: x[1]["avg"], reverse=True)
    most_exp      = ranked[0]
    cheapest      = ranked[-1]
    gap           = most_exp[1]["avg"] - cheapest[1]["avg"]

    ranking_str = " > ".join(f"{d} (LKR {s['avg']:,})" for d, s in ranked)

    DISTRICT_REASONS = {
        "Colombo":      "high land scarcity, proximity to the CBD, business districts, and international schools",
        "Negombo":      "coastal location, tourist infrastructure, and proximity to the international airport",
        "Galle":        "coastal lifestyle, colonial heritage, and growing expat demand",
        "Kandy":        "hill country desirability, cultural significance, and steady demand from the central province",
        "Jaffna":       "post-war development and improving infrastructure driving gradual price growth",
        "Matara":       "southern coastal growth corridor with improving highway access",
        "Kurunegala":   "North Western Province hub with agricultural land base keeping prices moderate",
        "Ratnapura":    "gem-mining economy with moderate urbanisation",
        "Anuradhapura": "historical significance but lower urbanisation keeping land prices moderate",
        "Badulla":      "remote hill country location with limited commercial development",
    }

    lines = [f"District price analysis [{src}]:"]
    lines.append(
        f"{most_exp[0]} is the most expensive district at an average of LKR {most_exp[1]['avg']:,} "
        f"({most_exp[1]['count']} properties), which is "
        f"{most_exp[1]['avg'] / overall_avg:.1f}x the overall average of LKR {overall_avg:,}. "
        f"This is driven by {DISTRICT_REASONS.get(most_exp[0], 'high demand and limited supply')}."
    )
    lines.append(
        f"{cheapest[0]} is the most affordable at LKR {cheapest[1]['avg']:,} "
        f"({cheapest[1]['count']} properties) — "
        f"{most_exp[1]['avg'] / max(cheapest[1]['avg'], 1):.1f}x cheaper than {most_exp[0]}. "
        f"The price gap between cheapest and most expensive district is LKR {gap:,}."
    )
    for d, s in ranked[1:-1]:
        reason = DISTRICT_REASONS.get(d, "regional demand factors")
        lines.append(f"{d} averages LKR {s['avg']:,} ({s['count']} properties) — {reason}.")
    lines.append(f"Full district price ranking: {ranking_str}.")

    return {"text": " ".join(lines), "source": f"{src}:district_analysis", "type": "analysis"}


def _property_type_doc(stats: dict, src: str) -> dict | None:
    pt_stats    = stats.get("property_type_stats", {})
    overall_avg = stats.get("overall_avg", 0)
    if not pt_stats:
        return None

    ranked = sorted(pt_stats.items(), key=lambda x: x[1]["avg"], reverse=True)

    TYPE_REASONS = {
        "Villa":        "premium build quality, larger land plots, private gardens, and often a pool or garage",
        "Land & House": "combined land and construction value, typically on larger plots",
        "House":        "standard residential construction on moderate land size",
        "Apartment":    "no land ownership, shared facilities, but lower entry price and easy maintenance",
    }

    lines = [f"Property type price analysis [{src}]:"]
    for pt, s in ranked:
        reason = TYPE_REASONS.get(pt, "property-specific factors")
        lines.append(
            f"{pt}: average LKR {s['avg']:,} ({s['count']} properties) — {reason}."
        )
    most_exp = ranked[0]
    cheapest = ranked[-1]
    lines.append(
        f"The premium between the most expensive ({most_exp[0]}) and most affordable "
        f"({cheapest[0]}) property type is LKR {most_exp[1]['avg'] - cheapest[1]['avg']:,}."
    )

    return {"text": " ".join(lines), "source": f"{src}:property_type_analysis", "type": "analysis"}


def _bedroom_scaling_doc(stats: dict, pred_df, src: str) -> dict | None:
    bedroom_stats = stats.get("bedroom_stats", {})
    if not bedroom_stats:
        return None

    buckets   = [(k, v) for k, v in bedroom_stats.items() if v > 0]
    avg_price = int(pred_df["estimated_price"].mean())

    lines = [f"How bedroom count affects price [{src}]:"]
    prev = None
    for bucket, avg_cost in buckets:
        if prev:
            pct = (avg_cost - prev) / prev * 100
            lines.append(f"{bucket}: LKR {avg_cost:,} average (up {pct:.0f}% vs previous tier).")
        else:
            lines.append(f"{bucket}: LKR {avg_cost:,} average.")
        prev = avg_cost

    lines.append(
        f"Each additional bedroom typically adds significant value through increased floor area "
        f"and the premium buyers place on flexibility. "
        f"The overall dataset average is LKR {avg_price:,}."
    )

    return {"text": " ".join(lines), "source": f"{src}:bedroom_scaling", "type": "analysis"}


def _land_size_doc(stats: dict, src: str) -> dict | None:
    land_stats = stats.get("land_stats", {})
    if not land_stats:
        return None

    buckets = [(k, v) for k, v in land_stats.items() if v > 0]

    lines = [f"How land size affects price [{src}]:"]
    prev = None
    for bucket, avg_cost in buckets:
        if prev:
            pct = (avg_cost - prev) / prev * 100
            lines.append(f"{bucket}: LKR {avg_cost:,} average (up {pct:.0f}% vs previous tier).")
        else:
            lines.append(f"{bucket}: LKR {avg_cost:,} average.")
        prev = avg_cost

    lines.append(
        f"Land size (in perches) is one of the most direct price drivers in Sri Lanka "
        f"because land is bought and sold separately from construction. "
        f"In high-demand districts like Colombo, each additional perch can add LKR 800,000+ to value."
    )

    return {"text": " ".join(lines), "source": f"{src}:land_size_analysis", "type": "analysis"}


def _age_analysis_doc(stats: dict, src: str) -> dict | None:
    age_stats = stats.get("age_stats", {})
    if not age_stats:
        return None

    buckets = [(k, v) for k, v in age_stats.items() if v > 0]

    lines = [f"How property age affects price [{src}]:"]
    for bucket, avg_cost in buckets:
        lines.append(f"{bucket}: LKR {avg_cost:,} average.")

    if len(buckets) >= 2:
        newest_cost = list(age_stats.values())[0]
        oldest_cost = list(age_stats.values())[-1]
        diff = newest_cost - oldest_cost
        lines.append(
            f"New properties (0-5 years) command a premium of approximately LKR {diff:,} "
            f"over older properties (20+ years). "
            f"This reflects modern finishes, better energy efficiency, and lower maintenance costs. "
            f"Older properties can still command premium prices if renovated or in prime locations."
        )

    return {"text": " ".join(lines), "source": f"{src}:age_analysis", "type": "analysis"}


def _cost_drivers_doc(stats: dict, src: str) -> dict | None:
    importance  = stats.get("grouped_importance", {})
    if not importance:
        return None

    ranked = list(importance.items())[:7]
    total  = sum(v for _, v in importance.items())

    lines = [f"Key price drivers based on ML model [{src}]:"]
    lines.append(
        "The Random Forest model identified these factors as most important for predicting house price:"
    )
    for i, (feat, imp) in enumerate(ranked, 1):
        pct   = imp / total * 100 if total > 0 else 0
        label = _feat_label(feat)
        lines.append(f"{i}. {label} — {pct:.1f}% of model's predictive power (score: {imp:.4f}).")

    top   = _feat_label(ranked[0][0])
    top2  = _feat_label(ranked[1][0]) if len(ranked) > 1 else ""
    dist_pct = importance.get("district (all)", 0) / total * 100 if total > 0 else 0
    lines.append(
        f"In plain terms: {top} is the single biggest lever for house price, "
        f"followed by {top2}. "
        f"Location (district) alone explains {dist_pct:.0f}% of price variation — "
        f"confirming that 'location, location, location' holds true in the Sri Lankan market."
    )

    return {"text": " ".join(lines), "source": f"{src}:cost_drivers", "type": "analysis"}


def _premium_features_doc(stats: dict, pred_df, src: str) -> dict | None:
    lines = [f"Impact of premium features on price [{src}]:"]

    if "has_pool" in pred_df.columns:
        pool_avg    = int(pred_df[pred_df["has_pool"] == 1]["estimated_price"].mean()) if pred_df["has_pool"].sum() > 0 else 0
        no_pool_avg = int(pred_df[pred_df["has_pool"] == 0]["estimated_price"].mean()) if (pred_df["has_pool"] == 0).sum() > 0 else 0
        if pool_avg and no_pool_avg:
            lines.append(
                f"Properties with a swimming pool average LKR {pool_avg:,} vs "
                f"LKR {no_pool_avg:,} without — a premium of LKR {pool_avg - no_pool_avg:,}."
            )

    if "has_garage" in pred_df.columns:
        garage_avg    = int(pred_df[pred_df["has_garage"] == 1]["estimated_price"].mean()) if pred_df["has_garage"].sum() > 0 else 0
        no_garage_avg = int(pred_df[pred_df["has_garage"] == 0]["estimated_price"].mean()) if (pred_df["has_garage"] == 0).sum() > 0 else 0
        if garage_avg and no_garage_avg:
            lines.append(
                f"Properties with a garage average LKR {garage_avg:,} vs "
                f"LKR {no_garage_avg:,} without — a premium of LKR {garage_avg - no_garage_avg:,}."
            )

    if "furnished" in pred_df.columns:
        furn_avg    = int(pred_df[pred_df["furnished"] == 1]["estimated_price"].mean()) if pred_df["furnished"].sum() > 0 else 0
        unfurn_avg  = int(pred_df[pred_df["furnished"] == 0]["estimated_price"].mean()) if (pred_df["furnished"] == 0).sum() > 0 else 0
        if furn_avg and unfurn_avg:
            lines.append(
                f"Furnished properties average LKR {furn_avg:,} vs "
                f"LKR {unfurn_avg:,} unfurnished — a premium of LKR {furn_avg - unfurn_avg:,}."
            )

    if len(lines) == 1:
        return None

    return {"text": " ".join(lines), "source": f"{src}:premium_features", "type": "analysis"}


def _dataset_overview_doc(pipeline_results: dict, src: str) -> dict:
    s = pipeline_results["dataset_stats"]

    dist_breakdown = " | ".join(
        f"{d}: LKR {p:,}"
        for d, p in sorted(s.get("district_avg_prices", {}).items(), key=lambda x: -x[1])
    )
    type_breakdown = " | ".join(
        f"{t}: LKR {p:,}"
        for t, p in sorted(s.get("property_type_avg_prices", {}).items(), key=lambda x: -x[1])
    )

    return {
        "text": (
            f"Dataset Overview [{src}]: {s['total_properties']} properties. "
            f"Price range: LKR {s['min_estimated_price']:,} to LKR {s['max_estimated_price']:,}. "
            f"Average estimated price: LKR {s['avg_estimated_price']:,}. "
            f"Average bedrooms: {s['avg_bedrooms']}. "
            f"Districts: {', '.join(s['districts'])}. "
            f"Average price by district — {dist_breakdown}. "
            f"Average price by property type — {type_breakdown}."
        ),
        "source": f"{src}:summary",
        "type":   "summary",
    }


def _model_metrics_doc(metrics: dict, src: str) -> dict | None:
    if not metrics:
        return None
    return {
        "text": (
            f"Model Performance [{src}]: Random Forest trained on {metrics['train_size']} properties, "
            f"tested on {metrics['test_size']}. "
            f"R-squared: {metrics['r2']} ({metrics['r2'] * 100:.1f}% of price variance explained). "
            f"Mean Absolute Error: LKR {metrics['mae']:,.0f}. "
            f"RMSE: LKR {metrics['rmse']:,.0f}. "
            f"Accuracy: {metrics['accuracy']:.1f}% (avg error: {metrics['mape']:.1f}%)."
        ),
        "source": f"{src}:metrics",
        "type":   "metrics",
    }


# ── Public entry point ─────────────────────────────────────────────────────────

def build_documents(pipeline_results: dict, source_name: str = "") -> list[dict]:
    src              = source_name or "dataset"
    pred_df          = pipeline_results["predictions_df"]
    metrics          = pipeline_results["metrics"]
    analytical_stats = pipeline_results.get("analytical_stats", {})

    documents = []

    for builder in [_district_analysis_doc, _property_type_doc,
                    _cost_drivers_doc]:
        doc = builder(analytical_stats, src)
        if doc:
            documents.append(doc)

    for builder in [_bedroom_scaling_doc, _land_size_doc,
                    _age_analysis_doc, _premium_features_doc]:
        doc = builder(analytical_stats, pred_df, src) if builder in [_bedroom_scaling_doc, _premium_features_doc] else builder(analytical_stats, src)
        if doc:
            documents.append(doc)

    documents.append(_dataset_overview_doc(pipeline_results, src))

    doc = _model_metrics_doc(metrics, src)
    if doc:
        documents.append(doc)

    return documents


def embed_text(text: str) -> list[float]:
    return _get_embedder().encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _get_embedder().encode(texts, normalize_embeddings=True, batch_size=64).tolist()


def load_and_embed(pipeline_results: dict, source_name: str = "", progress_callback=None) -> list[dict]:
    documents = build_documents(pipeline_results, source_name=source_name)
    total     = len(documents)

    if progress_callback:
        progress_callback(0, total)

    all_embeddings = []
    for start in range(0, total, 64):
        batch_texts = [d["text"] for d in documents[start: start + 64]]
        batch_embs  = embed_texts(batch_texts)
        all_embeddings.extend(batch_embs)
        if progress_callback:
            progress_callback(min(start + 64, total), total)

    for doc, emb in zip(documents, all_embeddings):
        doc["embedding"] = emb

    return documents
