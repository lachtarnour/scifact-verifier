"""Build positive/negative SciFact examples for dense retriever fine-tuning."""

import argparse
import json
import random
from pathlib import Path

import torch

from src.data.load_scifact import load_corpus, load_qrels, load_queries
from src.retrieval.bm25_retriever import BM25Retriever

from .config import RETRIEVER_CONFIG, RetrieverFinetuningConfig
from .encoder import DOCUMENT_ROLE, QUERY_ROLE, encode_texts, format_document, load_model


def validate_qrels(
    split: str,
    qrels: dict[str, dict[str, int]],
    queries: dict[str, str],
    corpus: dict[str, dict[str, str]],
) -> None:
    missing_queries = [query_id for query_id in qrels if query_id not in queries]
    missing_docs = [
        doc_id
        for doc_scores in qrels.values()
        for doc_id in doc_scores
        if doc_id not in corpus
    ]

    if missing_queries or missing_docs:
        raise ValueError(
            f"{split}: {len(missing_queries)} missing queries, "
            f"{len(missing_docs)} missing docs"
        )


def count_pairs(qrels: dict[str, dict[str, int]]) -> int:
    return sum(len(doc_scores) for doc_scores in qrels.values())


def split_train_validation(
    qrels: dict[str, dict[str, int]],
    validation_ratio: float,
    random_seed: int,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    query_ids = sorted(qrels, key=int)
    if len(query_ids) < 2:
        raise ValueError("At least two training claims are required to create a validation split.")

    shuffled_query_ids = query_ids[:]
    random.Random(random_seed).shuffle(shuffled_query_ids)
    validation_count = round(len(shuffled_query_ids) * validation_ratio)
    validation_count = max(1, min(len(shuffled_query_ids) - 1, validation_count))
    validation_query_ids = set(shuffled_query_ids[:validation_count])

    train_qrels = {
        query_id: qrels[query_id]
        for query_id in query_ids
        if query_id not in validation_query_ids
    }
    validation_qrels = {
        query_id: qrels[query_id]
        for query_id in query_ids
        if query_id in validation_query_ids
    }
    return train_qrels, validation_qrels


def rank_dense_negatives(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    settings: RetrieverFinetuningConfig,
) -> dict[str, list[str]]:
    tokenizer, model, device = load_model(
        base_model=settings.base_model,
        query_adapter=settings.query_adapter,
        document_adapter=settings.document_adapter,
    )
    sep_token = tokenizer.sep_token or "[SEP]"

    doc_ids = list(corpus)
    doc_texts = [
        format_document(corpus[doc_id].get("title", ""), corpus[doc_id].get("text", ""), sep_token)
        for doc_id in doc_ids
    ]
    query_ids = list(qrels)
    query_texts = [queries[query_id] for query_id in query_ids]

    doc_embeddings = encode_texts(
        doc_texts,
        tokenizer,
        model,
        device,
        settings.mining_batch_size,
        settings.max_length,
        role=DOCUMENT_ROLE,
        show_progress=True,
        description="Encoding documents for negatives",
    )
    query_embeddings = encode_texts(
        query_texts,
        tokenizer,
        model,
        device,
        settings.mining_batch_size,
        settings.max_length,
        role=QUERY_ROLE,
        show_progress=True,
        description="Encoding claims for negatives",
    )
    scores = query_embeddings @ doc_embeddings.T

    dense_negatives: dict[str, list[str]] = {}
    for row_index, query_id in enumerate(query_ids):
        positive_doc_ids = set(qrels[query_id])
        ranking = torch.argsort(scores[row_index], descending=True).tolist()
        dense_negatives[query_id] = [
            doc_ids[index]
            for index in ranking
            if doc_ids[index] not in positive_doc_ids
        ][:settings.dense_pool_size]

    return dense_negatives


def pick_negatives(
    query: str,
    positive_doc_ids: set[str],
    bm25: BM25Retriever,
    dense_doc_ids: list[str],
    all_doc_ids: list[str],
    rng: random.Random,
    num_negatives: int,
    settings: RetrieverFinetuningConfig,
) -> list[str]:
    bm25_doc_ids = [
        doc_id
        for doc_id in bm25.retrieve(query, top_k=settings.bm25_pool_size)
        if doc_id not in positive_doc_ids
    ]

    candidates: list[str] = []
    seen = set(positive_doc_ids)
    for bm25_doc_id, dense_doc_id in zip(bm25_doc_ids, dense_doc_ids):
        for doc_id in (bm25_doc_id, dense_doc_id):
            if doc_id not in seen:
                candidates.append(doc_id)
                seen.add(doc_id)
        if len(candidates) >= num_negatives:
            return candidates[:num_negatives]

    for doc_id in bm25_doc_ids + dense_doc_ids:
        if doc_id not in seen:
            candidates.append(doc_id)
            seen.add(doc_id)

    if len(candidates) < num_negatives:
        fallback = [
            doc_id
            for doc_id in all_doc_ids
            if doc_id not in seen
        ]
        rng.shuffle(fallback)
        candidates.extend(fallback)

    return candidates[:num_negatives]


def write_split(
    output_file: Path,
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    bm25: BM25Retriever,
    dense_negatives: dict[str, list[str]],
    num_negatives: int,
    random_seed: int,
    settings: RetrieverFinetuningConfig,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_doc_ids = list(corpus)
    rng = random.Random(random_seed)
    rows = 0

    with output_file.open("w", encoding="utf-8") as f:
        for query_id in sorted(qrels, key=int):
            doc_scores = qrels[query_id]
            query = queries[query_id]
            positive_doc_ids = set(doc_scores)
            negative_doc_ids = pick_negatives(
                query=query,
                positive_doc_ids=positive_doc_ids,
                bm25=bm25,
                dense_doc_ids=dense_negatives.get(query_id, []),
                all_doc_ids=all_doc_ids,
                rng=rng,
                num_negatives=num_negatives,
                settings=settings,
            )

            for positive_doc_id in sorted(positive_doc_ids, key=int):
                example = {
                    "query_id": query_id,
                    "query": query,
                    "positive_doc_id": positive_doc_id,
                    "negative_doc_ids": negative_doc_ids,
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                rows += 1

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=RETRIEVER_CONFIG.train_file)
    parser.add_argument("--validation-file", type=Path, default=RETRIEVER_CONFIG.validation_file)
    parser.add_argument("--test-file", type=Path, default=RETRIEVER_CONFIG.test_file)
    parser.add_argument("--validation-ratio", type=float, default=RETRIEVER_CONFIG.validation_ratio)
    parser.add_argument("--num-negatives", type=int, default=RETRIEVER_CONFIG.num_negatives)
    parser.add_argument("--seed", type=int, default=RETRIEVER_CONFIG.random_seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus = load_corpus()
    queries = load_queries()
    qrels_train_full = load_qrels("train")
    qrels_test = load_qrels("test")

    validate_qrels("train", qrels_train_full, queries, corpus)
    validate_qrels("test", qrels_test, queries, corpus)
    qrels_train, qrels_validation = split_train_validation(
        qrels=qrels_train_full,
        validation_ratio=args.validation_ratio,
        random_seed=args.seed,
    )

    bm25 = BM25Retriever()
    if not bm25.load():
        bm25.build(corpus)

    print(f"Ranking dense negatives with {RETRIEVER_CONFIG.base_model}")
    dense_negatives = rank_dense_negatives(
        corpus=corpus,
        queries=queries,
        qrels={**qrels_train, **qrels_validation, **qrels_test},
        settings=RETRIEVER_CONFIG,
    )

    train_rows = write_split(
        output_file=args.train_file,
        corpus=corpus,
        queries=queries,
        qrels=qrels_train,
        bm25=bm25,
        dense_negatives=dense_negatives,
        num_negatives=args.num_negatives,
        random_seed=args.seed,
        settings=RETRIEVER_CONFIG,
    )
    validation_rows = write_split(
        output_file=args.validation_file,
        corpus=corpus,
        queries=queries,
        qrels=qrels_validation,
        bm25=bm25,
        dense_negatives=dense_negatives,
        num_negatives=args.num_negatives,
        random_seed=args.seed,
        settings=RETRIEVER_CONFIG,
    )
    test_rows = write_split(
        output_file=args.test_file,
        corpus=corpus,
        queries=queries,
        qrels=qrels_test,
        bm25=bm25,
        dense_negatives=dense_negatives,
        num_negatives=args.num_negatives,
        random_seed=args.seed,
        settings=RETRIEVER_CONFIG,
    )

    print("Retriever fine-tuning data ready")
    print(
        f"- source train qrels: {len(qrels_train_full)} claims, "
        f"{count_pairs(qrels_train_full)} pairs"
    )
    print(f"- train: {len(qrels_train)} claims, {count_pairs(qrels_train)} pairs, {train_rows} examples")
    print(
        f"- validation: {len(qrels_validation)} claims, "
        f"{count_pairs(qrels_validation)} pairs, {validation_rows} examples"
    )
    print(f"- test: {len(qrels_test)} claims, {count_pairs(qrels_test)} pairs, {test_rows} examples")
    print(f"- negatives per example: {args.num_negatives}")


if __name__ == "__main__":
    main()
