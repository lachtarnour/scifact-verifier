"""Evaluate a SPECTER2 retriever on SciFact qrels."""

import argparse

import torch

from src.data.load_scifact import load_corpus, load_qrels, load_queries
from src.evaluation.metrics import compute_all_metrics

from .config import RETRIEVER_CONFIG
from .encoder import DOCUMENT_ROLE, QUERY_ROLE, encode_texts, format_document, load_model


def rank_queries(
    doc_ids: list[str],
    doc_embeddings: torch.Tensor,
    query_ids: list[str],
    query_texts: list[str],
    tokenizer,
    model,
    device,
    batch_size: int,
    max_length: int,
    show_progress: bool,
) -> dict[str, dict[str, float]]:
    query_embeddings = encode_texts(
        query_texts,
        tokenizer,
        model,
        device,
        batch_size,
        max_length,
        role=QUERY_ROLE,
        show_progress=show_progress,
        description="Encoding queries",
    )
    scores = query_embeddings @ doc_embeddings.T

    results: dict[str, dict[str, float]] = {}
    max_rank = min(100, len(doc_ids))
    for row_index, query_id in enumerate(query_ids):
        ranking = torch.argsort(scores[row_index], descending=True).tolist()[:max_rank]
        results[query_id] = {
            doc_ids[index]: float(scores[row_index, index])
            for index in ranking
        }

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=RETRIEVER_CONFIG.eval_batch_size)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = None if args.device == "auto" else torch.device(args.device)
    tokenizer, model, device = load_model(
        base_model=RETRIEVER_CONFIG.base_model,
        query_adapter=RETRIEVER_CONFIG.query_adapter,
        document_adapter=RETRIEVER_CONFIG.document_adapter,
        lora_adapter=args.lora_adapter,
        device=device,
    )
    model.eval()

    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels(args.split)

    sep_token = tokenizer.sep_token or "[SEP]"
    doc_ids = list(corpus)
    doc_texts = [
        format_document(corpus[doc_id].get("title", ""), corpus[doc_id].get("text", ""), sep_token)
        for doc_id in doc_ids
    ]
    qrels = {query_id: scores for query_id, scores in qrels.items() if query_id in queries}
    query_ids = list(qrels)
    query_texts = [queries[query_id] for query_id in query_ids]

    doc_embeddings = encode_texts(
        doc_texts,
        tokenizer,
        model,
        device,
        args.batch_size,
        RETRIEVER_CONFIG.max_length,
        role=DOCUMENT_ROLE,
        show_progress=not args.no_progress,
        description="Encoding documents",
    )
    results = rank_queries(
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        query_ids=query_ids,
        query_texts=query_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=RETRIEVER_CONFIG.max_length,
        show_progress=not args.no_progress,
    )
    metrics = compute_all_metrics(qrels, results)

    name = RETRIEVER_CONFIG.base_model
    if args.lora_adapter:
        name = f"{name} + {args.lora_adapter}"
    print(f"Retriever evaluation: {name}")
    print(f"- split: {args.split}")
    print(f"- queries: {len(query_ids)}")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
