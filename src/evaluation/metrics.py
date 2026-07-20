"""
IR evaluation metrics: Recall@k, Precision@k, MRR, nDCG@k.

All functions accept:
  qrels   : {query_id: {doc_id: relevance_score}}
  results : {query_id: {doc_id: retrieval_score}}
"""

import math


QrelsType = dict[str, dict[str, int]]
ResultsType = dict[str, dict[str, float]]


def ranked_doc_ids(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=scores.get, reverse=True)


def recall_at_k(qrels: QrelsType, results: ResultsType, k: int = 5) -> float:
    scores = []
    for qid, relevant in qrels.items():
        if qid not in results:
            scores.append(0.0)
            continue
        ranked = ranked_doc_ids(results[qid])
        top_k = set(ranked[:k])
        relevant_set = {d for d, r in relevant.items() if r > 0}
        if not relevant_set:
            continue
        scores.append(len(top_k & relevant_set) / len(relevant_set))
    return sum(scores) / len(scores) if scores else 0.0


def precision_at_k(qrels: QrelsType, results: ResultsType, k: int = 5) -> float:
    scores = []
    for qid, relevant in qrels.items():
        if qid not in results:
            scores.append(0.0)
            continue
        ranked = ranked_doc_ids(results[qid])
        top_k = ranked[:k]
        relevant_set = {d for d, r in relevant.items() if r > 0}
        hits = sum(1 for d in top_k if d in relevant_set)
        scores.append(hits / k)
    return sum(scores) / len(scores) if scores else 0.0


def mean_reciprocal_rank(qrels: QrelsType, results: ResultsType) -> float:
    scores = []
    for qid, relevant in qrels.items():
        if qid not in results:
            scores.append(0.0)
            continue
        ranked = ranked_doc_ids(results[qid])
        relevant_set = {d for d, r in relevant.items() if r > 0}
        rr = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant_set:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(qrels: QrelsType, results: ResultsType, k: int = 10) -> float:
    def dcg(ranked_rels: list) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(ranked_rels))

    scores = []
    for qid, relevant in qrels.items():
        if qid not in results:
            scores.append(0.0)
            continue
        ranked = ranked_doc_ids(results[qid])
        ranked_rels = [relevant.get(d, 0) for d in ranked[:k]]
        ideal_rels = sorted(relevant.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_rels)
        if ideal_dcg == 0:
            continue
        scores.append(dcg(ranked_rels) / ideal_dcg)
    return sum(scores) / len(scores) if scores else 0.0


def compute_all_metrics(
    qrels: QrelsType, results: ResultsType
) -> dict[str, float]:
    return {
        "Recall@1":  round(recall_at_k(qrels, results, k=1),  4),
        "Recall@5":  round(recall_at_k(qrels, results, k=5),  4),
        "Recall@10": round(recall_at_k(qrels, results, k=10), 4),
        "P@5":       round(precision_at_k(qrels, results, k=5), 4),
        "MRR":       round(mean_reciprocal_rank(qrels, results), 4),
        "nDCG@10":   round(ndcg_at_k(qrels, results, k=10), 4),
    }
