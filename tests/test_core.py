"""Core tests — one per component, no redundancy."""

import json
import pickle
import random
import sys
import types

import pytest
from flask import Flask
from unittest.mock import patch, MagicMock

from src.config import TokenizerConfig
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
from src.evaluation.metrics import compute_all_metrics
from src.rag.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from src.rag.factory import create_generator
from src.rag.ollama_generator import OllamaGenerator
from src.rag.base_generator import BaseGenerator


# ── Retrieval ────────────────────────────────────────────────────

def test_bm25_build_and_retrieve(tiny_corpus):
    bm25 = BM25Retriever(scenario_name="test", persist=False)
    bm25.build(tiny_corpus)
    results = bm25.retrieve("statins atrial fibrillation", top_k=3)
    assert len(results) == 3
    assert max(results, key=results.get) == "doc1"


def test_bm25_loads_old_tokenizer_config_pickle(tmp_path, monkeypatch):
    old_module_name = "src.utils.config"
    old_module = types.ModuleType(old_module_name)
    old_module.TokenizerConfig = TokenizerConfig

    original_module = TokenizerConfig.__module__
    sys.modules[old_module_name] = old_module
    TokenizerConfig.__module__ = old_module_name
    try:
        payload = {
            "bm25": object(),
            "doc_ids": ["doc1"],
            "tokenizer_config": TokenizerConfig(),
            "scenario_name": "old",
        }
        with open(tmp_path / "bm25_old.pkl", "wb") as f:
            pickle.dump(payload, f)
    finally:
        TokenizerConfig.__module__ = original_module
        sys.modules.pop(old_module_name, None)

    monkeypatch.setattr("src.retrieval.bm25_retriever.config.INDEX_DIR", tmp_path)
    bm25 = BM25Retriever(scenario_name="old")

    assert bm25.load()
    assert bm25.doc_ids == ["doc1"]
    assert isinstance(bm25.tokenizer_config, TokenizerConfig)


def test_rrf_fuses_results():
    r1 = {"doc1": 0.9, "doc2": 0.6}
    r2 = {"doc2": 0.95, "doc1": 0.5}
    fused = reciprocal_rank_fusion([r1, r2])
    assert fused["doc2"] >= fused["doc1"]


def test_dense_retriever_uses_specter2_config():
    dense = DenseRetriever()
    assert dense.model_name.endswith("specter2_base")
    assert dense.query_adapter.endswith("specter2_adhoc_query")
    assert dense.document_adapter.endswith("specter2")


# ── Fine-tuning ──────────────────────────────────────────────────

def test_format_document_uses_separator():
    from src.finetuning.retriever.encoder import format_document

    assert format_document("Title", "Abstract", "[SEP]") == "Title [SEP] Abstract"
    assert format_document("", "Abstract", "[SEP]") == "Abstract"


def test_pick_negatives_combines_bm25_dense_and_fallback():
    from src.finetuning.retriever.config import RetrieverFinetuningConfig
    from src.finetuning.retriever.prepare_data import pick_negatives

    class FakeBM25:
        def retrieve(self, query, top_k):
            assert query == "claim"
            assert top_k == 3
            return {"positive": 10.0, "bm25": 9.0, "shared": 8.0}

    result = pick_negatives(
        query="claim",
        positive_doc_ids={"positive"},
        bm25=FakeBM25(),
        dense_doc_ids=["dense", "shared"],
        all_doc_ids=["positive", "bm25", "dense", "shared", "fallback"],
        rng=random.Random(0),
        num_negatives=4,
        settings=RetrieverFinetuningConfig(bm25_pool_size=3),
    )

    assert result == ["bm25", "dense", "shared", "fallback"]


def test_split_train_validation_keeps_queries_disjoint():
    from src.finetuning.retriever.prepare_data import split_train_validation

    qrels = {
        str(index): {f"doc{index}": 1}
        for index in range(10)
    }

    train_qrels, validation_qrels = split_train_validation(
        qrels=qrels,
        validation_ratio=0.2,
        random_seed=13,
    )

    assert len(validation_qrels) == 2
    assert set(train_qrels).isdisjoint(validation_qrels)
    assert set(train_qrels) | set(validation_qrels) == set(qrels)


def test_retriever_dataset_builds_training_rows(tmp_path, tiny_corpus):
    from src.finetuning.retriever.training import RetrieverDataset, collate_batch, load_eval_examples

    train_file = tmp_path / "train.jsonl"
    row = {
        "query": "Statins reduce atrial fibrillation.",
        "positive_doc_id": "doc1",
        "negative_doc_ids": ["doc2", "doc3"],
    }
    train_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = RetrieverDataset(train_file, tiny_corpus, "[SEP]")
    item = dataset[0]
    batch = collate_batch([item])

    assert len(dataset) == 1
    assert item["query"] == "Statins reduce atrial fibrillation."
    assert item["positive"].startswith("Statins and atrial fibrillation [SEP]")
    assert len(item["negatives"]) == 2
    assert batch["queries"] == ["Statins reduce atrial fibrillation."]

    eval_file = tmp_path / "validation.jsonl"
    rows = [
        {"query_id": "q1", "query": "claim", "positive_doc_id": "doc1", "negative_doc_ids": []},
        {"query_id": "q1", "query": "claim", "positive_doc_id": "doc2", "negative_doc_ids": []},
    ]
    eval_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    queries, qrels = load_eval_examples(eval_file)
    assert queries == {"q1": "claim"}
    assert qrels == {"q1": {"doc1": 1, "doc2": 1}}


def test_lora_targets_only_attention_modules():
    import re

    from src.finetuning.retriever.config import LORA_CONFIG

    pattern = re.compile(LORA_CONFIG.target_modules)

    assert pattern.fullmatch("bert.encoder.layer.0.attention.self.query")
    assert pattern.fullmatch("bert.encoder.layer.11.attention.self.value")
    assert not pattern.fullmatch("bert.encoder.layer.0.output.adapters.query")


# ── API ──────────────────────────────────────────────────────────

def test_random_claim_endpoint(monkeypatch):
    from app.routes import bp

    monkeypatch.setattr(
        "app.routes.load_queries",
        lambda: {"q1": "Vitamin D supplementation reduces respiratory infections."},
    )

    app = Flask(__name__)
    app.register_blueprint(bp)
    response = app.test_client().get("/api/random-claim")

    assert response.status_code == 200
    assert response.get_json() == {
        "id": "q1",
        "claim": "Vitamin D supplementation reduces respiratory infections.",
    }


# ── Metrics ──────────────────────────────────────────────────────

def test_compute_all_metrics(tiny_qrels, tiny_results):
    metrics = compute_all_metrics(tiny_qrels, tiny_results)
    assert set(metrics.keys()) == {"Recall@1", "Recall@5", "Recall@10", "P@5", "MRR", "nDCG@10"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


# ── RAG ──────────────────────────────────────────────────────────

def test_prompt_builder(tiny_corpus):
    prompt = build_user_prompt("Statins reduce AF.", ["doc1", "doc2"], tiny_corpus)
    assert "Statins reduce AF." in prompt
    assert "[Document 1]" in prompt
    assert "[Document 2]" in prompt


@patch("src.rag.factory.config")
def test_factory_selects_ollama_by_default(mock_config):
    mock_config.ANTHROPIC_API_KEY = ""
    mock_config.OLLAMA_MODEL = "mistral"
    mock_config.OLLAMA_URL = "http://localhost:11434"
    gen = create_generator()
    assert isinstance(gen, OllamaGenerator)


@patch("src.rag.ollama_generator.requests.post")
def test_ollama_generate(mock_post, tiny_corpus):
    payload = {
        "verdict": "SUPPORTED",
        "confidence": 0.8,
        "explanation": "The evidence supports the claim.",
        "cited_docs": [1],
        "evidence": ["Statins reduce atrial fibrillation."],
    }
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": json.dumps(payload)}}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    gen = OllamaGenerator(model="mistral", base_url="http://localhost:11434")
    result = gen.generate("Statins reduce AF.", ["doc1"], tiny_corpus)
    assert result["verdict"] == "SUPPORTED"
    assert result["cited_docs"] == [1]
