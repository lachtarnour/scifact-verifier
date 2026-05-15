# SciFact RAG Verifier

Simple RAG application for scientific claim verification on the SciFact dataset.

The system retrieves biomedical abstracts, reranks them, and uses a local LLM to return a structured verdict:

```text
SUPPORTED / REFUTED / NOT ENOUGH INFO
```

# Interface
![SciFact RAG Verifier Interface](app/images/image.png)

## Pipeline

```text
Claim → BM25 / Dense Retrieval → RRF Fusion → Reranking → LLM → Verdict
```

## Features

- BM25 retrieval
- FAISS-based dense retrieval with SPECTER
- RRF hybrid fusion
- Cross-encoder reranking
- Local LLM generation with Ollama
- JSON verdict output
- Basic post-processing to reduce unsupported answers
- Flask interface with streaming responses

## Stack

Python, Flask, FAISS, sentence-transformers, rank-bm25, Ollama.

## Quick Start

```bash
pip install -r requirements.txt

ollama pull mistral
ollama serve

python setup_indexes.py
python run.py
```

Open:

```text
http://localhost:5000
```

## Configuration

Example `.env`:

```bash
OLLAMA_MODEL=mistral
OLLAMA_URL=http://localhost:11434
FLASK_DEBUG=False
```

## API

| Endpoint | Description |
|---|---|
| `/api/chat` | Retrieval + streamed generation |
| `/api/search` | Retrieval only |
| `/api/health` | Health check |
| `/api/metrics` | Retrieval metrics |

## Evaluation

```bash
python -m src.evaluation
```

Example results:

| Retriever | R@5 | nDCG | MRR |
|---|---:|---:|---:|
| BM25 | 0.739 | 0.674 | 0.634 |
| Dense | 0.402 | 0.356 | 0.314 |
| Hybrid | 0.659 | 0.580 | 0.523 |
| Hybrid + Reranker | 0.758 | 0.700 | 0.667 |

## Limitations

This is a practical prototype. Retrieved abstracts may be related without directly supporting the claim, so verdicts are post-processed and should not be interpreted as medical advice.