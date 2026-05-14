# ── Stage 1: Build (compile C extensions) ──────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ── Stage 2: Runtime (slim, no build tools) ─────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub
ENV SENTENCE_TRANSFORMERS_HOME=/cache/sentence_transformers
ENV FLASK_DEBUG=False
# Suppress noisy warnings
ENV HF_DATASETS_DISABLE_PROGRESS_BARS=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV TOKENIZERS_PARALLELISM=false

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data'); nltk.download('punkt_tab', download_dir='/usr/local/nltk_data')"

COPY . .

EXPOSE 5000

CMD ["sh", "-c", "python setup_indexes.py && gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 600 run:app"]
