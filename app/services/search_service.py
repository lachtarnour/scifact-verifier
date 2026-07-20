"""
Search service: retrieval + LLM generation with post-processing.
"""

import json
import time
from collections.abc import Generator

from app.services.retrieval_service import RetrievalService
from src.rag import create_generator
from src.config import config
from src.utils import get_logger

logger = get_logger(__name__)

_generator = None


def _get_generator():
    global _generator
    if _generator is None:
        _generator = create_generator()
    return _generator


def sanitize_verdict(raw: str) -> dict:
    """Parse JSON from LLM output and enforce consistency."""
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 0.0,
            "explanation": raw.strip()[:300],
            "cited_docs": [],
            "evidence": [],
        }

    verdict = parsed.get("verdict", "").strip().upper()
    valid = {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}

    if verdict not in valid:
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": 0.0,
            "explanation": parsed.get("explanation", "Invalid verdict from model."),
            "cited_docs": [],
            "evidence": [],
        }

    parsed["verdict"] = verdict

    if verdict in {"SUPPORTED", "REFUTED"} and not parsed.get("cited_docs"):
        return {
            "verdict": "NOT ENOUGH INFO",
            "confidence": min(float(parsed.get("confidence", 0.0)), 0.3),
            "explanation": "The retrieved abstracts do not provide direct evidence.",
            "cited_docs": [],
            "evidence": [],
        }

    return parsed


def search(
    claim: str,
    mode: str = "hybrid_rerank",
    top_k: int = 10,
) -> dict:
    svc = RetrievalService.get()
    t0 = time.time()

    doc_ids, scores = svc.retrieve(claim, mode=mode, top_k=top_k)
    formatted_docs = _format_docs(doc_ids, svc.corpus, scores)
    elapsed_ms = round((time.time() - t0) * 1000)

    return {
        "claim": claim,
        "mode": mode,
        "elapsed_ms": elapsed_ms,
        "num_results": len(doc_ids),
        "documents": formatted_docs,
    }


def stream_generation(
    claim: str,
    mode: str = "hybrid_rerank",
    top_k: int | None = None,
    model: str | None = None,
) -> Generator[str, None, None]:
    svc = RetrievalService.get()
    gen = _get_generator()

    if model and hasattr(gen, "model"):
        gen.model = model

    logger.info("Using LLM model: %s", getattr(gen, "model", "unknown"))

    k = top_k if top_k is not None else config.LLM_TOP_K
    doc_ids, scores = svc.retrieve(claim, mode=mode, top_k=k)
    formatted_docs = _format_docs(doc_ids, svc.corpus, scores)

    retrieval_data = json.dumps({
        "type": "retrieval",
        "documents": formatted_docs,
        "mode": mode,
    })
    yield f"data: {retrieval_data}\n\n"

    full_response = ""
    for token in gen.stream(claim, doc_ids, svc.corpus):
        full_response += token
        token_data = json.dumps({"type": "token", "content": token})
        yield f"data: {token_data}\n\n"

    verdict = sanitize_verdict(full_response)
    yield f"data: {json.dumps({'type': 'verdict', 'verdict': verdict})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


def _format_docs(
    doc_ids: list[str],
    corpus: dict,
    scores: dict[str, float],
) -> list[dict]:
    result = []
    for i, doc_id in enumerate(doc_ids, start=1):
        doc = corpus.get(doc_id, {})
        result.append({
            "rank": i,
            "doc_id": doc_id,
            "title": doc.get("title", "Unknown"),
            "abstract": doc.get("text", ""),
            "score": round(scores.get(doc_id, 0.0), 4),
        })
    return result
