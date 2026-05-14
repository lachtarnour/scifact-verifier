"""
Cross-encoder reranker.

Takes the top-N candidates from the hybrid retriever and re-scores each
(claim, abstract) pair with a cross-encoder.  Cross-encoders are much slower
than bi-encoders (no precomputed index) but significantly more accurate
because they see the query and document together.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — fast and accurate.
"""

from typing import Dict, List

from sentence_transformers import CrossEncoder

from src.data.load_scifact import CorpusType
from src.data.preprocess import build_doc_text
from src.retrieval.base_retriever import BaseRetriever
from src.utils import config, get_logger

logger = get_logger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.RERANKER_MODEL
        self._model: CrossEncoder | None = None

    def _load(self) -> None:
        if self._model is None:
            logger.info("Loading cross-encoder: %s …", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: Dict[str, float],
        corpus: CorpusType,
        top_k: int | None = None,
    ) -> Dict[str, float]:
        """
        Re-rank candidate doc_ids with the cross-encoder.
        Returns top_k doc_ids with their cross-encoder scores.
        """
        self._load()
        k = top_k or config.TOP_K_RERANK

        if not candidates:
            return {}

        pairs = []
        doc_id_list = list(candidates.keys())
        for doc_id in doc_id_list:
            doc = corpus.get(doc_id, {})
            doc_text = build_doc_text(doc.get("title", ""), doc.get("text", ""))
            pairs.append([query, doc_text])

        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(doc_id_list, scores.tolist()), key=lambda x: x[1], reverse=True
        )
        return {doc_id: float(score) for doc_id, score in ranked[:k]}



class RerankedRetriever(BaseRetriever):
    def __init__(
        self,
        base_retriever,
        corpus,
        reranker = Reranker(),
        candidate_k: int = 50,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.corpus = corpus
        self.candidate_k = candidate_k

    def load(self) -> bool:
        return self.base_retriever.load()
    

    def retrieve(self, query: str, top_k: int | None = None) -> Dict[str, float]:
        candidates = self.base_retriever.retrieve(
            query,
            top_k=self.candidate_k,
        )

        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            corpus=self.corpus,
            top_k=top_k
        )