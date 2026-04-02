
import io
import os
import json
import asyncio
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from rag_ml_runner   import train_and_save, predict_with_saved_model, model_exists
from rag_data_loader import load_and_embed
from rag_vector_db   import HouseVectorDB
from rag_engine      import answer_question

CACHE_DIR     = "data"
CACHE_CSV     = os.path.join(CACHE_DIR, "_pipeline_predictions.csv")
CACHE_META    = os.path.join(CACHE_DIR, "_pipeline_meta.json")
CACHE_SOURCES = os.path.join(CACHE_DIR, "_sources.json")

app = FastAPI(title="House Price RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ──────────────────────────────────────────────────────────────
_db: Optional[HouseVectorDB] = None
_pipeline_res: Optional[dict] = None
_sources: list[str] = []


def get_db() -> HouseVectorDB:
    global _db
    if _db is None:
        _db = HouseVectorDB()
    return _db


def _save_cache(pipeline_res: dict, sources: list):
    os.makedirs(CACHE_DIR, exist_ok=True)
    pipeline_res["predictions_df"].to_csv(CACHE_CSV, index=False)
    meta = {
        "metrics":            pipeline_res["metrics"],
        "feature_importance": pipeline_res["feature_importance"],
        "dataset_stats":      pipeline_res["dataset_stats"],
        "features":           pipeline_res["features"],
    }
    with open(CACHE_META, "w") as f:
        json.dump(meta, f)
    with open(CACHE_SOURCES, "w") as f:
        json.dump(sources, f)


def _load_cache():
    global _pipeline_res, _sources
    try:
        pred_df = pd.read_csv(CACHE_CSV)
        with open(CACHE_META) as f:
            meta = json.load(f)
        with open(CACHE_SOURCES) as f:
            _sources = json.load(f)
        _pipeline_res = {
            "predictions_df":     pred_df,
            "metrics":            meta["metrics"],
            "feature_importance": meta["feature_importance"],
            "dataset_stats":      meta["dataset_stats"],
            "features":           meta["features"],
        }
    except Exception:
        pass


def _merge_pipeline_results(base: dict, new: dict) -> dict:
    merged_df = pd.concat(
        [base["predictions_df"], new["predictions_df"]],
        ignore_index=True,
    )

    cost_col      = "estimated_price"
    dist_col      = "district"      if "district"      in merged_df.columns else None
    prop_type_col = "property_type" if "property_type" in merged_df.columns else None

    merged_stats = {
        "total_properties":   len(merged_df),
        "avg_estimated_price": int(merged_df[cost_col].mean()),
        "min_estimated_price": int(merged_df[cost_col].min()),
        "max_estimated_price": int(merged_df[cost_col].max()),
        "avg_bedrooms":        int(merged_df["bedrooms"].mean()) if "bedrooms" in merged_df.columns else 0,
        "districts":           merged_df[dist_col].unique().tolist() if dist_col else [],
        "district_avg_prices": (
            merged_df.groupby(dist_col)[cost_col].mean().round(0).astype(int).to_dict()
        ) if dist_col else {},
        "property_type_avg_prices": (
            merged_df.groupby(prop_type_col)[cost_col].mean().round(0).astype(int).to_dict()
        ) if prop_type_col else {},
    }

    return {
        "predictions_df":     merged_df,
        "metrics":            new["metrics"],
        "feature_importance": new["feature_importance"],
        "dataset_stats":      merged_stats,
        "features":           new["features"],
    }


def _validate_csv(df: pd.DataFrame) -> list[str]:
    errors = []
    required = {"district", "property_type", "bedrooms", "bathrooms", "floors",
                 "land_perches", "floor_area_sqft", "age_years",
                 "has_garage", "has_pool", "furnished"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
    if len(df) < 5:
        errors.append(f"Need at least 5 rows, got {len(df)}.")
    return errors


# ── Startup: restore cache ────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    _load_cache()
    try:
        get_db()
    except Exception:
        pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    global _pipeline_res, _sources
    try:
        db = get_db()
        n  = db.count()
    except Exception:
        n = 0

    stats   = _pipeline_res["dataset_stats"] if _pipeline_res else {}
    metrics = _pipeline_res["metrics"]       if _pipeline_res else {}

    return {
        "indexed":       n > 0,
        "doc_count":     n,
        "sources":       _sources,
        "stats":         stats,
        "metrics":       metrics,
        "model_trained": model_exists(),
    }


def _inject_api_key(request: Request):
    key = request.headers.get("X-Api-Key", "")
    if key:
        os.environ["OPENROUTER_API_KEY"] = key


# ── Step 1: Train ─────────────────────────────────────────────────────────────

@app.post("/train")
async def train(request: Request, file: UploadFile = File(...)):
    """
    Accept a CSV with price_lkr, train the model, and immediately
    index analytical docs from all training data into Qdrant.
    """
    _inject_api_key(request)
    global _pipeline_res, _sources

    contents = await file.read()
    df_raw   = pd.read_csv(io.BytesIO(contents))

    errors = _validate_csv(df_raw)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    if "price_lkr" not in df_raw.columns:
        raise HTTPException(
            status_code=400,
            detail="Training file must have a 'price_lkr' column with actual prices."
        )

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, train_and_save, df_raw)

    # Index analytical docs from all training data into Qdrant
    pipeline_res = result["pipeline_results"]
    source_name  = file.filename

    docs = await loop.run_in_executor(
        None,
        lambda: load_and_embed(pipeline_res, source_name=source_name)
    )

    db = get_db()
    db.reset()          # clear any old data before reindexing
    n  = db.upsert(docs)

    _sources      = [source_name]
    _pipeline_res = pipeline_res
    _save_cache(_pipeline_res, _sources)

    return {
        "ok":            True,
        "filename":      file.filename,
        "train_rows":    result["train_rows"],
        "feature_count": result["feature_count"],
        "metrics":       result["metrics"],
        "indexed_docs":  n,
    }


# ── Step 2: Predict & Index ───────────────────────────────────────────────────

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    """
    Accept a CSV WITHOUT total_cost_lkr, predict costs using the saved model,
    and index the results into Qdrant for RAG.
    """
    _inject_api_key(request)
    global _pipeline_res, _sources

    if not model_exists():
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Upload a training CSV first."
        )

    contents = await file.read()
    df_raw   = pd.read_csv(io.BytesIO(contents))

    errors = _validate_csv(df_raw)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    loop         = asyncio.get_event_loop()
    pipeline_res = await loop.run_in_executor(None, predict_with_saved_model, df_raw)

    docs = await loop.run_in_executor(
        None,
        lambda: load_and_embed(pipeline_res, source_name=file.filename)
    )

    db = get_db()
    n  = db.upsert(docs)

    if file.filename not in _sources:
        _sources.append(file.filename)

    if _pipeline_res is not None:
        _pipeline_res = _merge_pipeline_results(_pipeline_res, pipeline_res)
    else:
        _pipeline_res = pipeline_res
    _save_cache(_pipeline_res, _sources)

    s = _pipeline_res["dataset_stats"]

    return {
        "indexed_docs": n,
        "filename":     file.filename,
        "stats":        s,
    }


class PredictRequest(BaseModel):
    district:       str
    property_type:  str
    bedrooms:       int
    bathrooms:      int
    floors:         int
    land_perches:   float
    floor_area_sqft: int
    age_years:      int
    has_garage:     int
    has_pool:       int
    furnished:      int


@app.post("/predict-single")
async def predict_single(req: PredictRequest, request: Request):
    """
    Accept a single event's details, predict its cost, index it into Qdrant for RAG.
    """
    _inject_api_key(request)
    global _pipeline_res, _sources

    if not model_exists():
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Train the model first."
        )

    df_row = pd.DataFrame([{
        "district":        req.district,
        "property_type":   req.property_type,
        "bedrooms":        req.bedrooms,
        "bathrooms":       req.bathrooms,
        "floors":          req.floors,
        "land_perches":    req.land_perches,
        "floor_area_sqft": req.floor_area_sqft,
        "age_years":       req.age_years,
        "has_garage":      req.has_garage,
        "has_pool":        req.has_pool,
        "furnished":       req.furnished,
    }])

    loop         = asyncio.get_event_loop()
    pipeline_res = await loop.run_in_executor(None, predict_with_saved_model, df_row)

    estimated_price = int(pipeline_res["predictions_df"]["estimated_price"].iloc[0])

    source_name = "user_inputs"
    docs = await loop.run_in_executor(
        None,
        lambda: load_and_embed(pipeline_res, source_name=source_name)
    )

    db = get_db()
    db.upsert(docs)

    if source_name not in _sources:
        _sources.append(source_name)

    if _pipeline_res is not None:
        _pipeline_res = _merge_pipeline_results(_pipeline_res, pipeline_res)
    else:
        _pipeline_res = pipeline_res
    _save_cache(_pipeline_res, _sources)

    return {
        "estimated_price": estimated_price,
        "stats":           _pipeline_res["dataset_stats"],
        "event": {
            "district":        req.district,
            "property_type":   req.property_type,
            "bedrooms":        req.bedrooms,
            "bathrooms":       req.bathrooms,
            "floors":          req.floors,
            "land_perches":    req.land_perches,
            "floor_area_sqft": req.floor_area_sqft,
            "age_years":       req.age_years,
            "has_garage":      req.has_garage,
            "has_pool":        req.has_pool,
            "furnished":       req.furnished,
        }
    }


class ChatRequest(BaseModel):
    question: str
    history:  list[dict] = []


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    _inject_api_key(request)
    try:
        db = get_db()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant not ready: {e}")

    if db.count() == 0:
        raise HTTPException(status_code=400, detail="No data indexed yet. Upload a CSV first.")

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: answer_question(req.question, db, top_k=6, chat_history=req.history)
    )

    return {
        "answer":     result["answer"],
        "contexts":   result["contexts"],
        "model_used": result["model_used"],
    }


@app.post("/reset")
def reset():
    global _pipeline_res, _sources
    db = get_db()
    db.reset()
    _pipeline_res = None
    _sources      = []
    for f in [CACHE_CSV, CACHE_META, CACHE_SOURCES]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    return {"ok": True}
