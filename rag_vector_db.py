"""
rag_vector_db.py — Qdrant vector store for event planning RAG
-------------------------------------------------------------
Dense retrieval (local sentence-transformers embeddings) + cross-encoder reranker.
Supports per-source deletion and optional source filtering in search.
"""

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import CrossEncoder

COLLECTION = "events"
EMBED_DIM  = 384    # all-MiniLM-L6-v2 (local, no API)

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


class HouseVectorDB:
    """Qdrant collection with dense retrieval + cross-encoder reranking."""

    def __init__(self, url: str | None = None):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=url)
        self._ensure_collection()

    # ── Collection lifecycle ──────────────────────────────────────────────────

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION in existing:
            info = self.client.get_collection(COLLECTION)
            current_dim = info.config.params.vectors.size
            if current_dim != EMBED_DIM:
                self.client.delete_collection(COLLECTION)
                existing.remove(COLLECTION)
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name=COLLECTION,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def reset(self):
        """Drop and recreate the entire collection."""
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        self._ensure_collection()

    def delete_by_source(self, source_name: str):
        """
        Delete all points belonging to a specific source file.
        Matches both the base name (e.g. 'file.csv') and suffixed variants
        (e.g. 'file.csv:summary', 'file.csv:metrics', 'file.csv:insights').
        """
        ids_to_delete = []
        offset = None
        while True:
            result, offset = self.client.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(
                    should=[
                        FieldCondition(key="source", match=MatchValue(value=source_name)),
                        FieldCondition(key="source", match=MatchValue(value=f"{source_name}:summary")),
                        FieldCondition(key="source", match=MatchValue(value=f"{source_name}:metrics")),
                        FieldCondition(key="source", match=MatchValue(value=f"{source_name}:insights")),
                        FieldCondition(key="source", match=MatchValue(value=f"{source_name}:city_analysis")),
                        FieldCondition(key="source", match=MatchValue(value=f"{source_name}:event_type_analysis")),
                        FieldCondition(key="source", match=MatchValue(value=f"{source_name}:guest_scaling")),
                        FieldCondition(key="source", match=MatchValue(value=f"{source_name}:cost_drivers")),
                        FieldCondition(key="source", match=MatchValue(value=f"{source_name}:venue_food_analysis")),
                        FieldCondition(key="source", match=MatchValue(value=f"{source_name}:season_entertainment_analysis")),
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids_to_delete.extend([p.id for p in result])
            if offset is None:
                break

        if ids_to_delete:
            self.client.delete(
                collection_name=COLLECTION,
                points_selector=ids_to_delete,
            )

        return len(ids_to_delete)

    def count(self) -> int:
        return self.client.count(COLLECTION).count

    def get_sources(self) -> list[str]:
        """Return unique base source file names stored in Qdrant payloads."""
        sources = set()
        offset  = None
        while True:
            result, offset = self.client.scroll(
                collection_name=COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in result:
                src  = point.payload.get("source", "")
                base = src.split(":")[0]
                if base and base not in ("events", "summary", "model_metrics",
                                         "feature_importance", "dataset", "unknown"):
                    sources.add(base)
            if offset is None:
                break
        return sorted(sources)

    def fetch_summary_docs(self) -> list[dict]:
        """Fetch all summary, metrics, and insights documents directly (no vector search needed)."""
        results = []
        offset  = None
        while True:
            result, offset = self.client.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(
                    should=[
                        FieldCondition(key="type", match=MatchValue(value="summary")),
                        FieldCondition(key="type", match=MatchValue(value="metrics")),
                        FieldCondition(key="type", match=MatchValue(value="insights")),
                        FieldCondition(key="type", match=MatchValue(value="analysis")),
                    ]
                ),
                limit=50,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in result:
                results.append({
                    "text":   point.payload["text"],
                    "type":   point.payload.get("type", "summary"),
                    "source": point.payload.get("source", ""),
                    "score":  1.0,
                })
            if offset is None:
                break
        return results

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert(self, documents: list[dict]) -> int:
        """
        documents: list of {text, embedding, source, type}
        Returns number of points upserted.
        """
        points = []
        for i, doc in enumerate(documents):
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{doc.get('source','?')}:{i}:{doc['text'][:40]}"
            ))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=doc["embedding"],
                    payload={
                        "text":   doc["text"],
                        "source": doc.get("source", "unknown"),
                        "type":   doc.get("type",   "event"),
                    },
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=COLLECTION,
                points=points[i : i + batch_size],
            )

        return len(points)

    # ── Search + rerank ───────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 6,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        1. Dense cosine similarity search (fetch top_k × 4 candidates)
        2. Cross-encoder reranking → return top_k

        source_filter: if provided, restrict search to documents from that source file.
        """
        fetch_k = max(top_k * 4, 20)

        payload_filter = None
        if source_filter:
            payload_filter = Filter(
                should=[
                    FieldCondition(key="source", match=MatchValue(value=source_filter)),
                    FieldCondition(key="source", match=MatchValue(value=f"{source_filter}:summary")),
                    FieldCondition(key="source", match=MatchValue(value=f"{source_filter}:metrics")),
                    FieldCondition(key="source", match=MatchValue(value=f"{source_filter}:insights")),
                ]
            )

        results = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            query_filter=payload_filter,
            limit=fetch_k,
            with_payload=True,
        )

        candidates = [
            {
                "text":   r.payload["text"],
                "type":   r.payload.get("type", "event"),
                "source": r.payload.get("source", ""),
                "score":  r.score,
            }
            for r in results.points
        ]

        # Cross-encoder reranking
        reranker = _get_reranker()
        pairs    = [[query_text, c["text"]] for c in candidates]
        scores   = reranker.predict(pairs)
        ranked   = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        return [c for _, c in ranked[:top_k]]
