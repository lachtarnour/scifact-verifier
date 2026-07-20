import pickle
from operator import itemgetter

from rank_bm25 import BM25Okapi

from src.data.load_scifact import CorpusType
from src.data.preprocess import get_flat_corpus, tokenize_for_bm25
from src.retrieval.base_retriever import BaseRetriever
from src.config import TokenizerConfig, config
from src.utils import get_logger

logger = get_logger(__name__)


class BM25IndexUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "src.utils.config" and name == "TokenizerConfig":
            return TokenizerConfig
        return super().find_class(module, name)


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig | None = None,
        scenario_name: str = "default",
        persist: bool = True,
    ):
        """
        Parameters
        ----------
        tokenizer_config : TokenizerConfig
            Controls stemming, stop words, etc.
            Defaults to config.tokenizer (the global default scenario).
        scenario_name : str
            Used as part of the index filename so each scenario has
            its own cached index on disk.
        persist : bool
            Whether build() should save the index to disk.
        """
        self.tokenizer_config = tokenizer_config or config.tokenizer
        self.scenario_name = scenario_name
        self.persist = persist
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []

    # ── Build / load ──────────────────────────────────────────────

    def build(self, corpus: CorpusType) -> None:
        doc_ids, doc_texts = get_flat_corpus(corpus)
        tokenized = [
            tokenize_for_bm25(text, self.tokenizer_config)
            for text in doc_texts
        ]
        self.bm25 = BM25Okapi(tokenized)
        self.doc_ids = doc_ids
        if self.persist:
            self._save()
        logger.info("BM25 built: %d docs, scenario=%s", len(doc_ids), self.scenario_name)

    def load(self) -> bool:
        path = config.INDEX_DIR / f"bm25_{self.scenario_name}.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = BM25IndexUnpickler(f).load()
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.tokenizer_config = data["tokenizer_config"]
        logger.info("BM25 loaded: %d docs", len(self.doc_ids))
        return True

    def _save(self) -> None:
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        path = config.INDEX_DIR / f"bm25_{self.scenario_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "doc_ids": self.doc_ids,
                    "tokenizer_config": self.tokenizer_config,
                    "scenario_name": self.scenario_name,
                },
                f,
            )

    # ── Retrieve ──────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> dict[str, float]:
        """Return {doc_id: bm25_score} for the top-k documents."""
        if self.bm25 is None:
            raise RuntimeError(
                f"BM25 index (scenario='{self.scenario_name}') not loaded. "
                "Call build() or load() first."
            )
        k = top_k or config.TOP_K_BM25
        # Use the stored tokenizer_config — guarantees index/query consistency
        tokens = tokenize_for_bm25(query, self.tokenizer_config)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.doc_ids, scores.tolist()),
            key=itemgetter(1),
            reverse=True,
        )
        return {doc_id: score for doc_id, score in ranked[:k]}
