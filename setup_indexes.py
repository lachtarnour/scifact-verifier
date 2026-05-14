"""
One-time setup: download SciFact data, build BM25 + FAISS indexes.

    python setup_indexes.py
    python setup_indexes.py --force
"""

import argparse
import sys
import time

import nltk

from src.data import load_corpus, load_queries, load_qrels
from src.retrieval import BM25Retriever, DenseRetriever
from src.utils import config, get_logger

logger = get_logger("setup")


def main(force: bool = False, split: str = "test") -> None:
    t0 = time.time()

    nltk.download("stopwords", quiet=True)

    corpus = load_corpus(force_reload=force)
    queries = load_queries(force_reload=force)
    qrels = load_qrels(split, force_reload=force)
    logger.info("Data: %d docs, %d queries, %d qrels", len(corpus), len(queries), len(qrels))

    bm25 = BM25Retriever()
    if not force and bm25.load():
        logger.info("BM25 index found, skipping")
    else:
        bm25.build(corpus)

    dense = DenseRetriever()
    if not force and dense.load():
        logger.info("Dense index found, skipping")
    else:
        dense.build(corpus)

    logger.info("Setup done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    try:
        main(force=args.force, split=args.split)
    except KeyboardInterrupt:
        sys.exit(1)
