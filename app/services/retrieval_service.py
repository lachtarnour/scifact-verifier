"""
Singleton retrieval service loaded once at Flask startup.

Holds the corpus and all retrievers in memory so that every request
avoids re-loading models.  If indexes don't exist yet, raises a clear
error telling the user to run setup_indexes.py first.
"""

from typing import Dict, List

from src.data import load_corpus
from src.retrieval import (
    BaseRetriever,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Reranker,
    RerankedRetriever,
)
from src.utils import config, get_logger

logger = get_logger(__name__)


class RetrievalService:
    _instance: "RetrievalService | None" = None

    def __init__(self):
        self.corpus: Dict = {}
        self.hybrid: HybridRetriever | None = None
        self.reranked: RerankedRetriever | None = None
        self._ready = False

    @classmethod
    def get(cls) -> "RetrievalService":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        logger.info("Loading corpus")
        self.corpus = load_corpus()

        # HybridRetriever.load() delegates to each sub-retriever
        self.hybrid = HybridRetriever([BM25Retriever(), DenseRetriever()])

        if not self.hybrid.load():
            raise RuntimeError(
                "Retrieval indexes not found. "
                "Please run:  python setup_indexes.py"
            )

        self.reranked = RerankedRetriever(
            base_retriever=self.hybrid,
            reranker=Reranker(),
            corpus=self.corpus,
        )
        self._ready = True
        logger.info("Ready — %d docs", len(self.corpus))

    @property
    def ready(self) -> bool:
        return self._ready

    _MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

    def _get_retriever(self, mode: str) -> "BaseRetriever":
        """Return the retriever instance matching *mode*."""
        mapping = {
            "bm25": self.hybrid.retrievers[0],      # BM25Retriever
            "dense": self.hybrid.retrievers[1],      # DenseRetriever
            "hybrid": self.hybrid,
            "hybrid_rerank": self.reranked,
        }
        retriever = mapping.get(mode)
        if retriever is None:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose from: {self._MODES}"
            )
        return retriever

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        top_k: int = 10,
    ) -> tuple[List[str], Dict[str, float]]:
        """
        Retrieve documents for a query.

        mode options: 'bm25' | 'dense' | 'hybrid' | 'hybrid_rerank'
        Returns (ordered_doc_ids, score_dict).
        """
        retriever = self._get_retriever(mode)
        scores = retriever.retrieve(query, top_k=top_k)

        doc_ids = list(scores.keys())
        return doc_ids, scores
