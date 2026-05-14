import os
import pickle
import time
import warnings
from typing import Dict, List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", message=".*cache_dir.*deprecated.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.data.load_scifact import CorpusType
from src.data.preprocess import get_flat_corpus
from src.retrieval.base_retriever import BaseRetriever
from src.utils import config, get_logger

logger = get_logger(__name__)

_FAISS_PATH = config.INDEX_DIR / "dense.faiss"
_META_PATH = config.INDEX_DIR / "dense_meta.pkl"


def _detect_device() -> str:
    """Return the best available device for sentence-transformers."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _optimal_batch_size(device: str) -> int:
    """
    Conservative defaults.

    MPS can be fast, but 256 is not always stable or optimal depending on
    model size and text length. 128 is usually a safer default.
    """
    return {
        "mps": 128,
        "cuda": 256,
        "cpu": 128,
    }[device]


class DenseRetriever(BaseRetriever):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.doc_ids: List[str] = []
        self._device = _detect_device()

    # ── Build / load ──────────────────────────────────────────────

    def _load_model(self) -> None:
        if self.model is not None:
            return

        logger.info("Embedding model: %s (%s)", self.model_name, self._device)

        self.model = SentenceTransformer(
            self.model_name,
            device=self._device,
            model_kwargs={"cache_dir": str(config.INDEX_DIR / "models")},
        )

    def build(self, corpus: CorpusType, batch_size: int | None = None) -> None:
        self._load_model()

        doc_ids, doc_texts = get_flat_corpus(corpus)

        if not doc_ids:
            raise ValueError("Cannot build dense index: corpus is empty.")

        n = len(doc_ids)
        bs = batch_size or _optimal_batch_size(self._device)

        logger.info("Encoding %d docs (batch=%d)", n, bs)

        t0 = time.time()

        total_batches = (n + bs - 1) // bs
        all_embeddings = []

        for i in range(0, n, bs):
            batch_num = i // bs + 1
            batch = doc_texts[i : i + bs]
            emb = self.model.encode(
                batch,
                batch_size=bs,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            all_embeddings.append(emb)
            if batch_num % 20 == 0 or batch_num == total_batches:
                pct = 100 * min(i + bs, n) // n
                logger.info("  %d%% (%d/%d)", pct, min(i + bs, n), n)

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        elapsed = time.time() - t0

        logger.info("Encoded in %.1fs", elapsed)

        if embeddings.ndim != 2:
            raise RuntimeError(f"Invalid embedding shape: {embeddings.shape}")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.doc_ids = doc_ids

        self._save()

        logger.info("Dense built: %d vectors, dim=%d", self.index.ntotal, dimension)

    def load(self) -> bool:
        if not _FAISS_PATH.exists() or not _META_PATH.exists():
            return False

        self.index = faiss.read_index(str(_FAISS_PATH))

        with open(_META_PATH, "rb") as f:
            meta = pickle.load(f)

        self.doc_ids = meta["doc_ids"]
        saved_model_name = meta.get("model_name")

        if saved_model_name is not None:
            self.model_name = saved_model_name

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

        faiss.write_index(self.index, str(_FAISS_PATH))

        with open(_META_PATH, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "model_name": self.model_name,
                    "dimension": self.index.d,
                    "ntotal": self.index.ntotal,
                },
                f,
            )

    # ── Retrieve ──────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> Dict[str, float]:
        """Return {doc_id: cosine_similarity} for the top-k documents."""
        if self.index is None:
            raise RuntimeError("Dense index not loaded. Call build() or load() first.")

        if not query.strip():
            return {}

        self._load_model()

        k = top_k or config.TOP_K_DENSE
        k = min(k, self.index.ntotal)

        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(q_emb, k)

        results: Dict[str, float] = {}

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            results[self.doc_ids[idx]] = float(score)

        return results

    def retrieve_many(
        self,
        queries: List[str],
        top_k: int | None = None,
        batch_size: int | None = None,
    ) -> List[Dict[str, float]]:
        """
        Retrieve top-k documents for multiple queries.

        Much faster than calling retrieve() repeatedly during evaluation.
        """
        if self.index is None:
            raise RuntimeError("Dense index not loaded. Call build() or load() first.")

        if not queries:
            return []

        self._load_model()

        k = top_k or config.TOP_K_DENSE
        k = min(k, self.index.ntotal)

        bs = batch_size or _optimal_batch_size(self._device)

        q_embs = self.model.encode(
            queries,
            batch_size=bs,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(q_embs, k)

        all_results: List[Dict[str, float]] = []

        for row_indices, row_scores in zip(indices, scores):
            results: Dict[str, float] = {}

            for idx, score in zip(row_indices, row_scores):
                if idx == -1:
                    continue

                results[self.doc_ids[idx]] = float(score)

            all_results.append(results)

        return all_results