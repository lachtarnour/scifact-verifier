"""Configuration defaults for retriever fine-tuning."""

from dataclasses import dataclass
from pathlib import Path


RETRIEVER_DATA_DIR = Path("data/processed/fine_tuning/retriever")


@dataclass(frozen=True)
class RetrieverFinetuningConfig:
    base_model: str = "allenai/specter2_base"
    query_adapter: str = "allenai/specter2_adhoc_query"
    document_adapter: str = "allenai/specter2"

    train_file: Path = RETRIEVER_DATA_DIR / "train.jsonl"
    validation_file: Path = RETRIEVER_DATA_DIR / "validation.jsonl"
    test_file: Path = RETRIEVER_DATA_DIR / "test.jsonl"

    num_negatives: int = 7
    validation_ratio: float = 0.2
    bm25_pool_size: int = 50
    dense_pool_size: int = 50
    mining_batch_size: int = 32

    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    temperature: float = 0.05
    gradient_accumulation_steps: int = 1

    eval_batch_size: int = 32
    max_length: int = 512
    random_seed: int = 13


@dataclass(frozen=True)
class LoraExperimentConfig:
    output_dir: Path = Path("models/retriever/scifact-lora")
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = r"bert\.encoder\.layer\.\d+\.attention\.self\.(query|value)"


RETRIEVER_CONFIG = RetrieverFinetuningConfig()
LORA_CONFIG = LoraExperimentConfig()
