import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TokenizerConfig:
    use_stemming: bool = True
    use_stop_words: bool = True
    preserve_scientific_terms: bool = True
    min_token_length: int = 2
    stemmer: str = "porter"


TOKENIZER_SCENARIOS: dict[str, TokenizerConfig] = {
    "baseline": TokenizerConfig(
        use_stemming=False,
        use_stop_words=False,
        preserve_scientific_terms=False,
        min_token_length=1,
    ),
    "stopwords_only": TokenizerConfig(
        use_stemming=False,
        use_stop_words=True,
        preserve_scientific_terms=False,
    ),
    "stemming_only": TokenizerConfig(
        use_stemming=True,
        use_stop_words=False,
        preserve_scientific_terms=False,
    ),
    "scientific": TokenizerConfig(
        use_stemming=False,
        use_stop_words=True,
        preserve_scientific_terms=True,
    ),
    "full": TokenizerConfig(
        use_stemming=True,
        use_stop_words=True,
        preserve_scientific_terms=True,
    ),
    "full_snowball": TokenizerConfig(
        use_stemming=True,
        use_stop_words=True,
        preserve_scientific_terms=True,
        stemmer="snowball",
    ),
}

DEFAULT_TOKENIZER_SCENARIO = "full"


@dataclass
class Config:
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    INDEX_DIR: Path = BASE_DIR / "data" / "indexes"
    REPORTS_DIR: Path = BASE_DIR / "reports"

    # Retrieval models
    DEVICE: str = os.getenv("DEVICE", "cpu")
    SPECTER2_BASE_MODEL: str = os.getenv("SPECTER2_BASE_MODEL", "allenai/specter2_base")
    SPECTER2_QUERY_ADAPTER: str = os.getenv("SPECTER2_QUERY_ADAPTER", "allenai/specter2_adhoc_query")
    SPECTER2_DOCUMENT_ADAPTER: str = os.getenv("SPECTER2_DOCUMENT_ADAPTER", "allenai/specter2")
    SPECTER2_LORA_ADAPTER: str = os.getenv("SPECTER2_LORA_ADAPTER", "")
    EMBEDDING_MAX_LENGTH: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
    DENSE_INDEX_BATCH_SIZE: int = int(os.getenv("DENSE_INDEX_BATCH_SIZE", "32"))
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LLM
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.1
    LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", "5"))

    # Retrieval
    TOP_K_BM25: int = 100
    TOP_K_DENSE: int = 100
    TOP_K_HYBRID: int = 20
    TOP_K_RERANK: int = 10
    RRF_K: int = 60

    # BM25 tokenizer
    tokenizer: TokenizerConfig = TOKENIZER_SCENARIOS[DEFAULT_TOKENIZER_SCENARIO]

    # Flask
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")


config = Config()
