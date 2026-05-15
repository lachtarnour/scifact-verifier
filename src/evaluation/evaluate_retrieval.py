import time
from typing import Dict

from tqdm import tqdm

from src.evaluation.metrics import compute_all_metrics
from src.retrieval.base_retriever import BaseRetriever


def evaluate_retrieval(
    name: str,
    retriever: BaseRetriever,
    queries: Dict[str, str],
    qrels: Dict,
    top_k: int = 10,
) -> Dict:
    """
    Evaluate any retriever exposing:
        retriever.retrieve(query, top_k) -> Dict[doc_id, score]
    """

    results = {}

    t0 = time.time()

    for qid, query in tqdm(queries.items(), desc=name, unit="query"):
        results[qid] = retriever.(query, top_k=top_k)

    elapsed = time.time() - t0

    metrics = compute_all_metrics(qrels, results)
    metrics["latency_ms_per_query"] = round(elapsed / len(queries) * 1000, 1)
    metrics["top_k"] = top_k

    return metrics