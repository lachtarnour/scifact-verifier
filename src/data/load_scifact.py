"""
SciFact dataset loader (BeIR format via HuggingFace).

Corpus   : ~5 183 paper abstracts
Queries  : ~1 109 scientific claims
Qrels    : relevance judgments
"""

import json

from src.config import config
from src.utils import get_logger

logger = get_logger(__name__)

CorpusType = dict[str, dict[str, str]]
QueriesType = dict[str, str]
QrelsType = dict[str, dict[str, int]]


def _load_corpus_from_hf() -> CorpusType:
    from datasets import load_dataset

    ds = load_dataset("BeIR/scifact", "corpus", split="corpus", trust_remote_code=False)
    corpus: CorpusType = {}
    for row in ds:
        corpus[str(row["_id"])] = {"title": row["title"], "text": row["text"]}
    logger.info("Corpus: %d documents", len(corpus))
    return corpus


def _load_queries_from_hf() -> QueriesType:
    from datasets import load_dataset

    ds = load_dataset("BeIR/scifact", "queries", split="queries", trust_remote_code=False)
    queries: QueriesType = {}
    for row in ds:
        queries[str(row["_id"])] = row["text"]
    logger.info("Queries: %d claims", len(queries))
    return queries


def _load_qrels_from_hf(split: str = "test") -> QrelsType:
    from datasets import load_dataset

    ds = load_dataset("BeIR/scifact-qrels", split=split, trust_remote_code=False)
    qrels: QrelsType = {}
    for row in ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        rel = int(row["score"])
        qrels.setdefault(qid, {})[did] = rel
    logger.info("Qrels (%s): %d queries", split, len(qrels))
    return qrels


def load_corpus(force_reload: bool = False) -> CorpusType:
    cache_path = config.PROCESSED_DIR / "corpus.json"
    if not force_reload and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    corpus = _load_corpus_from_hf()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(corpus, f)
    return corpus


def load_queries(force_reload: bool = False) -> QueriesType:
    cache_path = config.PROCESSED_DIR / "queries.json"
    if not force_reload and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    queries = _load_queries_from_hf()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(queries, f)
    return queries


def load_qrels(split: str = "test", force_reload: bool = False) -> QrelsType:
    cache_path = config.PROCESSED_DIR / f"qrels_{split}.json"
    if not force_reload and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    qrels = _load_qrels_from_hf(split)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(qrels, f)
    return qrels
