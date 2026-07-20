ARG TRAIN_PLATFORM=linux/amd64
FROM --platform=$TRAIN_PLATFORM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libopenblas0 \
    tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub
ENV SENTENCE_TRANSFORMERS_HOME=/cache/sentence_transformers
ENV HF_DATASETS_DISABLE_PROGRESS_BARS=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV TOKENIZERS_PARALLELISM=false
ENV DEVICE=cuda

COPY docker/training/requirements.txt ./requirements-train.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements-train.txt

COPY . .

RUN mkdir -p /workspace/run /cache/huggingface \
    && rm -rf /app/data \
    && ln -s /workspace/run/data /app/data \
    && chmod -R 777 /workspace /cache

CMD ["bash", "scripts/ovh/train_lora.sh"]
