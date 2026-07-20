import time

from tqdm import tqdm

from src.evaluation.metrics import compute_all_metrics
from src.retrieval.base_retriever import BaseRetriever


def evaluate_retrieval(
    name: str,
    retriever: BaseRetriever,
    queries: dict[str, str],
    qrels: dict,
    top_k: int = 10,
) -> dict:
    """
    Evaluate any retriever exposing:
        retriever.retrieve(query, top_k) -> dict[doc_id, score]
    Optionally uses:
        retriever.retrieve_many(queries, top_k) -> list[dict[doc_id, score]]
    """
    if not queries:
        raise ValueError("Cannot evaluate retrieval with an empty query set.")

    results = {}

    t0 = time.time()

    query_items = list(queries.items())
    if hasattr(retriever, "retrieve_many"):
        query_texts = [query for _, query in query_items]
        batch_results = retriever.retrieve_many(query_texts, top_k=top_k)
        results = {
            query_id: result
            for (query_id, _), result in zip(query_items, batch_results)
        }
    else:
        for qid, query in tqdm(query_items, desc=name, unit="query"):
            results[qid] = retriever.retrieve(query, top_k=top_k)

    elapsed = time.time() - t0

    metrics = compute_all_metrics(qrels, results)
    metrics["latency_ms_per_query"] = round(elapsed / len(queries) * 1000, 1)
    metrics["top_k"] = top_k

    return metrics
