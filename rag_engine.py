"""
rag_engine.py — HyDE + Dense Search + LLM Answer
-------------------------------------------------
Pipeline per question:
  1. HyDE  — LLM generates a hypothetical factual passage for better dense retrieval
  2. Embed — embed the hypothetical passage with the local model
  3. Search — dense search + cross-encoder rerank
  4. Answer — LLM answers using retrieved event context + conversation history
"""

import os
import time
import requests

from rag_data_loader import embed_text
from rag_vector_db   import HouseVectorDB

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

LLM_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-4b:free",
    "mistralai/mistral-7b-instruct:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
]


# ── LLM helper ────────────────────────────────────────────────────────────────

def _call_llm(messages: list[dict], model: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":       model,
        "messages":    messages,
        "temperature": 0.2,
        "max_tokens":  1024,
    }
    resp = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "choices" not in data:
        raise KeyError(f"No 'choices' in response: {data}")
    return data["choices"][0]["message"]["content"].strip()


def _call_llm_with_fallback(messages: list[dict]) -> tuple[str, str]:
    """Tries each model in order. On 429 waits before next. Returns (answer, model_name)."""
    last_error = None
    for i, model in enumerate(LLM_MODELS):
        try:
            return _call_llm(messages, model), model
        except requests.HTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code == 429:
                wait = 5 * (i + 1)   # 5s, 10s, 15s … increasing per attempt
                time.sleep(wait)
            continue
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All LLM models failed. Last error: {last_error}")


# ── Step 1: HyDE ──────────────────────────────────────────────────────────────

def generate_hyde(question: str) -> str:
    """
    Generate a Hypothetical Document Embedding passage.
    A short factual answer written as if it were in the dataset —
    this matches the vector space of indexed documents better than
    a raw question does.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Sri Lankan real estate analyst. "
                "Write a short (2-3 sentence) factual passage about house prices "
                "that would directly answer the following question. "
                "Use specific numbers in LKR, Sri Lankan district names, property types, "
                "bedroom counts, land sizes in perches, and floor areas in sqft. "
                "Do NOT say 'I don't know' — write a plausible answer."
            ),
        },
        {"role": "user", "content": question},
    ]
    answer, _ = _call_llm_with_fallback(messages)
    return answer


# ── Question classifier ───────────────────────────────────────────────────────

_AGGREGATION_KEYWORDS = {
    "cheapest", "expensive", "average", "avg", "most", "least", "highest",
    "lowest", "total", "compare", "comparison", "overall", "summary",
    "breakdown", "distribution", "how many", "all cities", "all events",
    "which city", "which event", "which venue", "best", "worst", "range",
}

_EXPLANATION_KEYWORDS = {
    "why", "how", "explain", "reason", "factor", "matter",
    "affect", "impact", "drive", "cause", "what makes", "influence",
    "difference", "between", "compare", "vs",
}

def _is_aggregation_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _AGGREGATION_KEYWORDS)

def _is_explanation_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _EXPLANATION_KEYWORDS)


# ── Full RAG pipeline ─────────────────────────────────────────────────────────

def answer_question(
    question: str,
    db: HouseVectorDB,
    top_k: int = 6,
    chat_history: list[dict] | None = None,
    source_filter: str | None = None,
) -> dict:
    """
    Improved RAG pipeline:
      1. HyDE  — generate hypothetical answer passage for better retrieval
      2. Embed — embed the HyDE passage
      3. Search — dense search + cross-encoder rerank (event chunks)
      4. Inject — always include summary/insights docs for aggregation questions
      5. Answer — structured LLM answer with labelled context
    """
    # ── Step 1: HyDE — embed a hypothetical answer, not the raw question ──────
    # This dramatically improves retrieval: "A wedding in Colombo for 200 guests
    # costs LKR 5M" matches stored event chunks far better than the question itself.
    try:
        hyde_passage = generate_hyde(question)
        search_vector = embed_text(hyde_passage)
    except Exception:
        # Fall back to plain question embedding if HyDE LLM call fails
        hyde_passage  = question
        search_vector = embed_text(question)

    # ── Step 2: Dense search + cross-encoder rerank (event rows) ─────────────
    event_chunks = db.search(
        search_vector,
        question,
        top_k=top_k,
        source_filter=source_filter,
    )

    # ── Step 3: Always inject summary/insights docs ───────────────────────────
    # Aggregation questions ("cheapest city?", "average cost?") need the
    # dataset overview doc — it may not rank in top_k from vector search alone.
    summary_chunks = db.fetch_summary_docs()

    # Always include summary; include analysis docs for all questions
    # (they are small in number and always relevant)
    summary_docs  = [c for c in summary_chunks if c["type"] == "summary"]
    analysis_docs = [c for c in summary_chunks if c["type"] == "analysis"]
    metrics_docs  = [c for c in summary_chunks if c["type"] in ("metrics", "insights")]

    event_texts = {c["text"] for c in event_chunks}
    extra_chunks = [c for c in summary_docs + analysis_docs if c["text"] not in event_texts]
    if _is_aggregation_question(question) or _is_explanation_question(question):
        extra_chunks += [c for c in metrics_docs if c["text"] not in event_texts]

    # ── Step 4: Build structured context block ────────────────────────────────
    # Label each chunk by type so the LLM knows what it's reading
    context_parts = []

    analysis_in_extra = [c for c in extra_chunks if c["type"] == "analysis"]
    summary_in_extra  = [c for c in extra_chunks if c["type"] == "summary"]
    other_in_extra    = [c for c in extra_chunks if c["type"] not in ("analysis", "summary")]

    if analysis_in_extra:
        context_parts.append("=== ANALYTICAL EXPLANATIONS (why/how costs differ) ===")
        for c in analysis_in_extra:
            context_parts.append(c["text"])

    if summary_in_extra:
        context_parts.append("=== DATASET STATISTICS ===")
        for c in summary_in_extra:
            context_parts.append(c["text"])

    if other_in_extra:
        context_parts.append("=== MODEL INFORMATION ===")
        for c in other_in_extra:
            context_parts.append(c["text"])

    if event_chunks:
        context_parts.append("=== INDIVIDUAL PROPERTY RECORDS ===")
        for i, c in enumerate(event_chunks):
            context_parts.append(f"[{i + 1}] {c['text']}")

    context_block = "\n\n".join(context_parts)

    # ── Step 5: Build prompt ──────────────────────────────────────────────────
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert Sri Lankan real estate advisor. "
            "You have access to two types of context:\n"
            "- ANALYTICAL EXPLANATIONS: ML-derived insights about why prices differ across districts, "
            "property types, land sizes, bedroom counts, age, and premium features.\n"
            "- DATASET STATISTICS: aggregate price averages by district and property type.\n\n"
            "Rules:\n"
            "- Use ONLY the numbers and explanations in the context — never invent figures.\n"
            "- For 'why' questions, draw on the analytical explanations and give a causal answer with LKR figures.\n"
            "- For 'how does X affect price' questions, use the scaling/driver analysis to show progression.\n"
            "- For 'what does X cost' questions, cite the statistics directly.\n"
            "- Format all prices as LKR X,XXX,XXX.\n"
            "- For comparisons, use a ranked list.\n"
            "- If the context lacks enough information, say so clearly."
        ),
    }

    history_messages = []
    if chat_history:
        for turn in chat_history[-6:]:
            history_messages.append({"role": turn["role"], "content": turn["content"]})

    user_msg = {
        "role": "user",
        "content": (
            f"Context:\n\n{context_block}\n\n"
            f"Question: {question}"
        ),
    }

    messages = [system_msg] + history_messages + [user_msg]
    answer, model_used = _call_llm_with_fallback(messages)

    all_contexts = [c["text"] for c in extra_chunks] + [c["text"] for c in event_chunks]

    return {
        "answer":     answer,
        "contexts":   all_contexts,
        "model_used": model_used,
    }
