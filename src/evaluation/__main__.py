"""
python -m src.evaluation
python -m src.evaluation --top-k 20 --split test
"""

import argparse
import json
import sys

from src.data import load_corpus, load_queries, load_qrels
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Reranker, RerankedRetriever
from src.config import config
from src.utils import get_logger

logger = get_logger("evaluation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels(args.split)
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    logger.info("%d queries, top_k=%d", len(queries), args.top_k)

    bm25 = BM25Retriever()
    if not bm25.load():
        bm25.build(corpus)

    dense = DenseRetriever()
    if not dense.load():
        dense.build(corpus, batch_size=config.DENSE_INDEX_BATCH_SIZE)

    hybrid = HybridRetriever([bm25, dense])
    reranked = RerankedRetriever(
        base_retriever=hybrid,
        reranker=Reranker(),
        corpus=corpus,
    )

    retrievers = {
        "BM25": bm25,
        "Dense": dense,
        "Hybrid": hybrid,
        "Hybrid+Reranker": reranked,
    }

    all_metrics = {}
    for name, retriever in retrievers.items():
        metrics = evaluate_retrieval(name, retriever, queries, qrels, top_k=args.top_k)
        all_metrics[name] = metrics
        logger.info(
            "%s — R@5=%.4f nDCG=%.4f MRR=%.4f %.1fms/q",
            name, metrics["Recall@5"], metrics["nDCG@10"],
            metrics["MRR"], metrics["latency_ms_per_query"],
        )

    out_path = args.output or str(config.REPORTS_DIR / "retrieval_metrics.json")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'Retriever':<20} {'R@5':>8} {'nDCG':>8} {'MRR':>8} {'ms/q':>8}")
    print("-" * 52)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['Recall@5']:>8.4f} {m['nDCG@10']:>8.4f} {m['MRR']:>8.4f} {m['latency_ms_per_query']:>8.1f}")


if __name__ == "__main__":
    main()
