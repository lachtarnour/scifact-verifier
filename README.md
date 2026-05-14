# SciFact RAG Verifier

Vérification de claims scientifiques par RAG sur le dataset [SciFact](https://github.com/allenai/scifact) (5 183 abstracts biomédicaux).

## Pipeline

```
Claim → BM25 + Dense (SPECTER) → RRF → Cross-encoder → LLM → Verdict
                                                              (SUPPORTED / REFUTED / NEI)
```

## Quick Start

### Local

```bash
pip install -r requirements.txt
ollama pull mistral && ollama serve &
python setup_indexes.py
python run.py
# → http://localhost:5000
```

### Docker

```bash
cp .env.example .env
docker compose up --build
```

Sur Mac Apple Silicon, Ollama tourne sur l'hôte (Metal) et le container l'atteint via `host.docker.internal`. Lance Ollama avec `OLLAMA_HOST=0.0.0.0:11434 ollama serve`.

## API

| Endpoint | Méthode | Description |
|---|---|---|
| `/api/chat` | POST | SSE streaming — retrieval + génération |
| `/api/search` | POST | Retrieval seul |
| `/api/health` | GET | Status |

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"claim": "Statins reduce atrial fibrillation risk."}'
```

## Évaluation

```bash
python -m src.evaluation
```

Compare BM25, Dense, Hybrid, Hybrid+Reranker (Recall@5, nDCG@10, MRR).

## Configuration

Variables d'environnement (`.env`) :

```bash
OLLAMA_MODEL=mistral
OLLAMA_URL=http://localhost:11434
ANTHROPIC_API_KEY=        # optionnel, bascule sur Claude
```

## Stack

Python 3.11, Flask, FAISS, sentence-transformers, rank-bm25, Ollama, Docker.
