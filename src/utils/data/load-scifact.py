"""
Load and cache the SciFact dataset in BEIR format.

The dataset is composed of:
- a corpus of scientific abstracts,
- scientific claims used as queries,
- qrels linking each claim to relevant documents.

In BEIR, queries are stored in a single split called "queries".
The train/test split is defined by the qrels.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

from datasets import load_dataset

from src.utils import config, get_logger

logger = get_logger(__name__)

CorpusType = Dict[str, Dict[str, str]]
QueriesType = Dict[str, str]
QrelsType = Dict[str, Dict[str, int]]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _load_corpus_from_hf() -> CorpusType:
    logger.info("Loading SciFact corpus from Hugging Face...")

    dataset = load_dataset(
        "BeIR/scifact",
        "corpus",
        split="corpus",
        trust_remote_code=True,
    )

    corpus = {
        str(row["_id"]): {
            "title": row["title"],
            "text": row["text"],
        }
        for row in dataset
    }

    logger.info("Loaded %d documents.", len(corpus))
    return corpus


def _load_queries_from_hf() -> QueriesType:
    logger.info("Loading SciFact queries from Hugging Face...")

    dataset = load_dataset(
        "BeIR/scifact",
        "queries",
        split="queries",
        trust_remote_code=True,
    )

    queries = {
        str(row["_id"]): row["text"]
        for row in dataset
    }

    logger.info("Loaded %d queries.", len(queries))
    return queries


def _load_qrels_from_hf(split: str = "test") -> QrelsType:
    logger.info("Loading SciFact qrels for split '%s'...", split)

    dataset = load_dataset(
        "BeIR/scifact-qrels",
        split=split,
        trust_remote_code=True,
    )

    qrels: QrelsType = {}

    for row in dataset:
        query_id = str(row["query-id"])
        document_id = str(row["corpus-id"])
        relevance = int(row["score"])

        qrels.setdefault(query_id, {})[document_id] = relevance

    logger.info("Loaded qrels for %d queries.", len(qrels))
    return qrels


def load_corpus(force_reload: bool = False) -> CorpusType:
    cache_path = config.PROCESSED_DIR / "corpus.json"

    if cache_path.exists() and not force_reload:
        logger.info("Loading corpus from cache.")
        return _read_json(cache_path)

    corpus = _load_corpus_from_hf()
    _write_json(corpus, cache_path)

    return corpus


def load_queries(force_reload: bool = False) -> QueriesType:
    cache_path = config.PROCESSED_DIR / "queries.json"

    if cache_path.exists() and not force_reload:
        logger.info("Loading queries from cache.")
        return _read_json(cache_path)

    queries = _load_queries_from_hf()
    _write_json(queries, cache_path)

    return queries


def load_qrels(split: str = "test", force_reload: bool = False) -> QrelsType:
    cache_path = config.PROCESSED_DIR / f"qrels_{split}.json"

    if cache_path.exists() and not force_reload:
        logger.info("Loading qrels for split '%s' from cache.", split)
        return _read_json(cache_path)

    qrels = _load_qrels_from_hf(split)
    _write_json(qrels, cache_path)

    return qrels


def load_all(split: str = "test") -> Tuple[CorpusType, QueriesType, QrelsType]:
    """
    Load the corpus, the qrels for a given split, and the matching queries.

    Since BEIR stores all queries together, we filter them using the query IDs
    present in the selected qrels split.
    """
    corpus = load_corpus()
    qrels = load_qrels(split)
    all_queries = load_queries()

    queries = {
        query_id: all_queries[query_id]
        for query_id in qrels
        if query_id in all_queries
    }

    logger.info(
        "Loaded split '%s': %d queries, %d qrels entries, %d corpus documents.",
        split,
        len(queries),
        sum(len(documents) for documents in qrels.values()),
        len(corpus),
    )

    return corpus, queries, qrels