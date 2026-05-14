"""Core tests — one per component, no redundancy."""

import pytest
from unittest.mock import patch, MagicMock

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
from src.evaluation.metrics import compute_all_metrics
from src.rag.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from src.rag.factory import create_generator
from src.rag.ollama_generator import OllamaGenerator
from src.rag.base_generator import BaseGenerator


# ── Retrieval ────────────────────────────────────────────────────

def test_bm25_build_and_retrieve(tiny_corpus):
    bm25 = BM25Retriever()
    bm25.build(tiny_corpus)
    results = bm25.retrieve("statins atrial fibrillation", top_k=3)
    assert len(results) == 3
    assert max(results, key=results.get) == "doc1"


def test_rrf_fuses_results():
    r1 = {"doc1": 0.9, "doc2": 0.6}
    r2 = {"doc2": 0.95, "doc1": 0.5}
    fused = reciprocal_rank_fusion([r1, r2])
    assert fused["doc2"] >= fused["doc1"]


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
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "SUPPORTED."}}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    gen = OllamaGenerator(model="mistral", base_url="http://localhost:11434")
    result = gen.generate("Statins reduce AF.", ["doc1"], tiny_corpus)
    assert result["content"] == "SUPPORTED."
