"""
Hybrid retriever combining multiple retrievers via Reciprocal Rank Fusion.

RRF(d) = Σ 1 / (k + rank_i(d))

RRF is parameter-light, robust to score scale differences between BM25 and
cosine similarity, and consistently outperforms simple weighted sums when
ranks are not on the same scale.

The retriever list is generic — any combination of BaseRetriever subclasses
can be fused (BM25 + Dense, Dense + Dense, etc.).
"""

from typing import Dict, List

from src.retrieval.base_retriever import BaseRetriever
from src.utils import config, get_logger

logger = get_logger(__name__)


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    results_list: List[Dict[str, float]],
    k: int | None = None,
) -> Dict[str, float]:
    """
    Fuse multiple ranked lists via RRF.
    Each dict maps doc_id → score (only ranking is used, not actual scores).
    Returns a dict of doc_id → fused_score.
    """
    rrf_k = k or config.RRF_K
    fused: Dict[str, float] = {}
    for results in results_list:
        ranked = sorted(results.keys(), key=lambda d: results[d], reverse=True)
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + _rrf_score(rank, rrf_k)
    return fused


class HybridRetriever(BaseRetriever):
    def __init__(self, retrievers: List[BaseRetriever]):
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever requires at least 2 retrievers.")
        self.retrievers = retrievers

    # ── Lifecycle — delegate to sub-retrievers ───────────────────

    def build(self, corpus, **kwargs) -> None:
        for r in self.retrievers:
            logger.info("Building %s …", type(r).__name__)
            r.build(corpus, **kwargs)

    def load(self) -> bool:
        results = []
        for r in self.retrievers:
            ok = r.load()
            results.append(ok)
            if not ok:
                logger.warning("%s failed to load.", type(r).__name__)
        return all(results)

    # ── Retrieve ─────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int | None = None, rrf_k: int | None = None
    ) -> Dict[str, float]:
        """Return top-k documents fused from all sub-retrievers via RRF."""
        k = top_k or config.TOP_K_HYBRID

        all_results = [
            r.retrieve(query, top_k=top_k) for r in self.retrievers
        ]

        fused = reciprocal_rank_fusion(all_results, k=rrf_k or config.RRF_K)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return {doc_id: score for doc_id, score in ranked[:k]}
