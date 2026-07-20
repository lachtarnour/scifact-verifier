"""
One-time setup: download SciFact data, build BM25 + FAISS indexes.

    python setup_indexes.py
    python setup_indexes.py --force
    python setup_indexes.py --force-dense
"""

import argparse
import sys
import time

import nltk

from src.data import load_corpus, load_queries, load_qrels
from src.retrieval import BM25Retriever, DenseRetriever
from src.config import config
from src.utils import get_logger

logger = get_logger("setup")

def main(
    force: bool = False,
    force_bm25: bool = False,
    force_dense: bool = False,
    split: str = "test",
) -> None:
    t0 = time.time()
    force_data = force or force_bm25 or force_dense

    nltk.download("stopwords", quiet=True)

    corpus = load_corpus(force_reload=force_data)
    queries = load_queries(force_reload=force_data)
    qrels = load_qrels(split, force_reload=force_data)
    logger.info("Data: %d docs, %d queries, %d qrels", len(corpus), len(queries), len(qrels))

    bm25 = BM25Retriever()
    if not (force or force_bm25) and bm25.load():
        logger.info("BM25 index found, skipping")
    else:
        bm25.build(corpus)

    dense = DenseRetriever()
    if not (force or force_dense) and dense.load():
        logger.info("Dense index found, skipping")
    else:
        dense.build(corpus, batch_size=config.DENSE_INDEX_BATCH_SIZE)

    logger.info("Setup done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-bm25", action="store_true")
    parser.add_argument("--force-dense", action="store_true")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    try:
        main(
            force=args.force,
            force_bm25=args.force_bm25,
            force_dense=args.force_dense,
            split=args.split,
        )
    except KeyboardInterrupt:
        sys.exit(1)
