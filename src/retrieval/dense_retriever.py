
import pickle 
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np 

from src.data.load_scifact import CorpusType
from src.data.preprocess import get_flat_corpus
from src.utils import Config, get_logger

logger = get_logger(__name__)

_FAISS_PATH = Config.INDEX_DIR / "dense.faiss"
_META_PATH = Config.INDEX_DIR / "dense_meta.pkl"
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.doc_ids: List[str] = []

    def _load_model(self) -> None:
        if self.model is None:
            logger.info("Loading embedding model: %s …", self.model_name)
            self.model = SentenceTransformer(self.model_name)

    def build(self, corpus: CorpusType, batch_size: int = 64) -> None:
        self._load_model()
        logger.info("Encoding %d documents with %s …", len(corpus), self.model_name)
        doc_ids, doc_texts = get_flat_corpus(corpus)

        embeddings = self.model.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine ≡ inner product after normalisation
            convert_to_numpy=True,
        ).astype(np.float32)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.doc_ids = doc_ids

        self._save()
        logger.info("Dense index built: %d vectors of dim %d.", len(doc_ids), dimension)

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
        Config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(_FAISS_PATH))
        with open(_META_PATH, "wb") as f:
            pickle.dump({"doc_ids": self.doc_ids, "model_name": self.model_name}, f)

    def retrieve(self, query: str, top_k: int | None = None) -> Dict[str, float]:
        """Return {doc_id: cosine_similarity} for the top-k documents."""
        if self.index is None:
            raise RuntimeError("Dense index not loaded. Call build() or load() first.")
        self._load_model()
        k = top_k or Config.TOP_K_DENSE

        q_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        scores, indices = self.index.search(q_emb, k)
        return {
            self.doc_ids[idx]: float(score)
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        }
