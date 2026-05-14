

import pickle
import time
from typing import Dict, List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data.load_scifact import CorpusType
from src.data.preprocess import get_flat_corpus
from src.utils import config, get_logger

logger = get_logger(__name__)

_FAISS_PATH = config.INDEX_DIR / "dense.faiss"
_META_PATH  = config.INDEX_DIR / "dense_meta.pkl"


def _detect_device() -> str:
    """Return the best available device for sentence-transformers."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _optimal_batch_size(device: str) -> int:
    """
    Larger batches saturate GPU/MPS pipelines and reduce per-sample overhead.
    M5 unified memory (16-32 GB) handles 256 comfortably for SPECTER (768 dim).
    """
    return {"mps": 256, "cuda": 256, "cpu": 64}[device]


class DenseRetriever:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.doc_ids: List[str] = []
        self._device = _detect_device()

    # ── Build / load ──────────────────────────────────────────────

    def _load_model(self) -> None:
        if self.model is None:
            logger.info(
                "Loading embedding model: %s (device=%s) …",
                self.model_name, self._device,
            )
            self.model = SentenceTransformer(self.model_name, device=self._device)

    def build(self, corpus: CorpusType, batch_size: int | None = None) -> None:
        self._load_model()
        doc_ids, doc_texts = get_flat_corpus(corpus)
        n = len(doc_ids)
        bs = batch_size or _optimal_batch_size(self._device)

        logger.info(
            "Encoding %d documents — model=%s, device=%s, batch_size=%d …",
            n, self.model_name, self._device, bs,
        )
        t0 = time.time()

        embeddings = self.model.encode(
            doc_texts,
            batch_size=bs,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine ≡ inner product after normalisation
            convert_to_numpy=True,
        ).astype(np.float32)

        elapsed = time.time() - t0
        logger.info(
            "Encoding done in %.1f s (%.1f docs/s).",
            elapsed, n / elapsed,
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.doc_ids = doc_ids

        self._save()
        logger.info("Dense index built: %d vectors of dim %d.", n, dimension)

    def load(self) -> bool:
        if not _FAISS_PATH.exists() or not _META_PATH.exists():
            logger.warning("No existing dense index found at %s.", _FAISS_PATH)
            return False
        logger.info("Loading FAISS index …")
        self.index = faiss.read_index(str(_FAISS_PATH))
        with open(_META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.doc_ids = meta["doc_ids"]
        self.model_name = meta.get("model_name", self.model_name)
        logger.info("Dense index loaded (%d vectors).", len(self.doc_ids))
        return True

    def _save(self) -> None:
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(_FAISS_PATH))
        with open(_META_PATH, "wb") as f:
            pickle.dump({"doc_ids": self.doc_ids, "model_name": self.model_name}, f)

    # ── Retrieve ──────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> Dict[str, float]:
        """Return {doc_id: cosine_similarity} for the top-k documents."""
        if self.index is None:
            raise RuntimeError("Dense index not loaded. Call build() or load() first.")
        self._load_model()
        k = top_k or config.TOP_K_DENSE

        q_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        scores, indices = self.index.search(q_emb, k)
        return {
            self.doc_ids[idx]: float(score)
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        }