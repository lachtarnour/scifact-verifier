import os
import pickle
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*cache_dir.*deprecated.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.data.load_scifact import CorpusType
from src.retrieval.base_retriever import BaseRetriever
from src.finetuning.retriever.encoder import DOCUMENT_ROLE, QUERY_ROLE
from src.finetuning.retriever.encoder import format_document
from src.finetuning.retriever.encoder import Specter2Encoder
from src.config import config
from src.utils import get_logger

logger = get_logger(__name__)

_FAISS_PATH = config.INDEX_DIR / "dense.faiss"
_META_PATH = config.INDEX_DIR / "dense_meta.pkl"


class DenseRetriever(BaseRetriever):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.SPECTER2_BASE_MODEL
        self.lora_adapter = config.SPECTER2_LORA_ADAPTER or None
        self.query_adapter = config.SPECTER2_QUERY_ADAPTER or None
        self.document_adapter = config.SPECTER2_DOCUMENT_ADAPTER or None
        self.model: Specter2Encoder | None = None
        self.index = None
        self.doc_ids: list[str] = []
        self._device = config.DEVICE.lower()
        if self._device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        elif self._device not in {"cpu", "cuda", "mps"}:
            raise ValueError("DEVICE must be one of: auto, cpu, cuda, mps")

    # ── Build / load ──────────────────────────────────────────────

    def _load_model(self) -> None:
        if self.model is not None:
            return

        logger.info("Dense model: %s (%s)", self.model_name, self._device)
        self.model = Specter2Encoder(
            base_model=self.model_name,
            query_adapter=self.query_adapter,
            document_adapter=self.document_adapter,
            lora_adapter=self.lora_adapter,
            device=torch.device(self._device),
            max_length=config.EMBEDDING_MAX_LENGTH,
        )

    def _corpus_texts(self, corpus: CorpusType) -> tuple[list[str], list[str]]:
        if self.model is None:
            raise RuntimeError("SPECTER2 model must be loaded before formatting documents.")

        doc_ids = []
        doc_texts = []
        sep_token = self.model.sep_token
        for doc_id, document in corpus.items():
            doc_ids.append(doc_id)
            doc_texts.append(
                format_document(
                    document.get("title", ""),
                    document.get("text", ""),
                    sep_token,
                )
            )
        return doc_ids, doc_texts

    def _encode(
        self,
        texts: list[str],
        batch_size: int,
        role: str | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded.")

        return self.model.encode(
            texts,
            batch_size=batch_size,
            role=role or DOCUMENT_ROLE,
        ).astype(np.float32)

    def _search(self, embeddings: np.ndarray, k: int) -> list[dict[str, float]]:
        if self.index is None:
            raise RuntimeError("Dense index not loaded. Call build() or load() first.")

        scores, indices = self.index.search(embeddings, k)
        all_results: list[dict[str, float]] = []

        for row_indices, row_scores in zip(indices, scores):
            results: dict[str, float] = {}
            for idx, score in zip(row_indices, row_scores):
                if idx != -1:
                    results[self.doc_ids[idx]] = float(score)
            all_results.append(results)

        return all_results

    def build(self, corpus: CorpusType, batch_size: int | None = None) -> None:
        self._load_model()
        doc_ids, doc_texts = self._corpus_texts(corpus)

        if not doc_ids:
            raise ValueError("Cannot build dense index: corpus is empty.")

        n = len(doc_ids)
        bs = batch_size or config.DENSE_INDEX_BATCH_SIZE

        logger.info("Encoding %d docs (batch=%d)", n, bs)

        t0 = time.time()

        total_batches = (n + bs - 1) // bs
        all_embeddings = []

        for i in range(0, n, bs):
            batch_num = i // bs + 1
            batch = doc_texts[i : i + bs]
            all_embeddings.append(self._encode(batch, batch_size=bs, role=DOCUMENT_ROLE))
            if batch_num % 20 == 0 or batch_num == total_batches:
                pct = 100 * min(i + bs, n) // n
                logger.info("  %d%% (%d/%d)", pct, min(i + bs, n), n)

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        elapsed = time.time() - t0

        logger.info("Encoded in %.1fs", elapsed)

        if embeddings.ndim != 2:
            raise RuntimeError(f"Invalid embedding shape: {embeddings.shape}")

        dimension = embeddings.shape[1]

        import faiss

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.doc_ids = doc_ids

        self._save()

        logger.info("Dense built: %d vectors, dim=%d", self.index.ntotal, dimension)

    def load(self) -> bool:
        if not _FAISS_PATH.exists() or not _META_PATH.exists():
            return False

        self._load_model()
        import faiss

        self.index = faiss.read_index(str(_FAISS_PATH))

        with open(_META_PATH, "rb") as f:
            meta = pickle.load(f)

        self.doc_ids = meta["doc_ids"]
        saved_metadata = {
            "model_name": meta.get("model_name"),
            "query_adapter": meta.get("query_adapter"),
            "document_adapter": meta.get("document_adapter"),
            "lora_adapter": meta.get("lora_adapter"),
        }
        expected_metadata = {
            "model_name": self.model_name,
            "query_adapter": self.query_adapter,
            "document_adapter": self.document_adapter,
            "lora_adapter": self.lora_adapter,
        }

        for key, expected_value in expected_metadata.items():
            saved_value = saved_metadata[key]
            if saved_value != expected_value:
                logger.warning(
                    "Dense index %s mismatch: index=%s config=%s",
                    key,
                    saved_value,
                    expected_value,
                )
                self.index = None
                self.doc_ids = []
                return False

        if self.index.ntotal != len(self.doc_ids):
            raise RuntimeError(
                f"Corrupted index: {self.index.ntotal} vectors vs {len(self.doc_ids)} doc_ids"
            )

        logger.info("Dense loaded: %d vectors", self.index.ntotal)

        return True

    def _save(self) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save: FAISS index is None.")

        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

        import faiss

        faiss.write_index(self.index, str(_FAISS_PATH))

        with open(_META_PATH, "wb") as f:
            metadata = {
                "doc_ids": self.doc_ids,
                "dimension": self.index.d,
                "ntotal": self.index.ntotal,
                "model_name": self.model_name,
                "query_adapter": self.query_adapter,
                "document_adapter": self.document_adapter,
                "lora_adapter": self.lora_adapter,
            }
            pickle.dump(metadata, f)

    # ── Retrieve ──────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> dict[str, float]:
        """Return {doc_id: cosine_similarity} for the top-k documents."""
        if self.index is None:
            raise RuntimeError("Dense index not loaded. Call build() or load() first.")

        if not query.strip():
            return {}

        self._load_model()

        k = top_k or config.TOP_K_DENSE
        k = min(k, self.index.ntotal)

        q_emb = self._encode([query], batch_size=1, role=QUERY_ROLE)
        return self._search(q_emb, k)[0]

    def retrieve_many(
        self,
        queries: list[str],
        top_k: int | None = None,
        batch_size: int | None = None,  # noqa: ARG002
    ) -> list[dict[str, float]]:
        """
        Retrieve top-k documents for multiple queries.
        """
        if not queries:
            return []

        return [self.retrieve(query, top_k=top_k) for query in queries]
