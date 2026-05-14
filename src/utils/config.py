import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict


BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ── Tokenizer configuration ───────────────────────────────────────────────────

@dataclass
class TokenizerConfig:
    """
    Controls how text is tokenized before BM25 indexing and querying.

    IMPORTANT: index and query must always use the same TokenizerConfig.
    Changing any parameter requires rebuilding the BM25 index.
    """
    use_stemming: bool = True
    # Porter stemmer — maps "reduces", "reduced", "reduction" → "reduc"

    use_stop_words: bool = True
    # Remove common English words + scientific filler words

    preserve_scientific_terms: bool = True
    # Keep hyphenated compounds intact: "IL-6", "COVID-19", "TNF-alpha"
    # Without this: "IL-6" → ["il", "6"] (two separate useless tokens)

    min_token_length: int = 2
    # Drop single-character tokens ("a", "b", "p", etc.)

    stemmer: str = "porter"
    # "porter"   — aggressive, fast (PorterStemmer)
    # "snowball"  — slightly more accurate, multi-language capable


# ── Predefined scenarios for evaluation ──────────────────────────────────────

TOKENIZER_SCENARIOS: Dict[str, TokenizerConfig] = {
    "baseline": TokenizerConfig(
        use_stemming=False,
        use_stop_words=False,
        preserve_scientific_terms=False,
        min_token_length=1,
    ),
    # Simple lowercase + split. Reference point.

    "stopwords_only": TokenizerConfig(
        use_stemming=False,
        use_stop_words=True,
        preserve_scientific_terms=False,
    ),
    # Adds stop word removal. Shows its isolated contribution.

    "stemming_only": TokenizerConfig(
        use_stemming=True,
        use_stop_words=False,
        preserve_scientific_terms=False,
    ),
    # Adds stemming only. Shows its isolated contribution.

    "scientific": TokenizerConfig(
        use_stemming=False,
        use_stop_words=True,
        preserve_scientific_terms=True,
    ),
    # Stop words + scientific term preservation, no stemming.
    # Good for corpora where exact term matching matters (gene names, drugs).

    "full": TokenizerConfig(
        use_stemming=True,
        use_stop_words=True,
        preserve_scientific_terms=True,
    ),
    # All features enabled. Expected best overall.

    "full_snowball": TokenizerConfig(
        use_stemming=True,
        use_stop_words=True,
        preserve_scientific_terms=True,
        stemmer="snowball",
    ),
    # Same as full but with Snowball stemmer instead of Porter.
    # Slightly less aggressive — compare with "full" to see the difference.
}

# The scenario used by default in the app and setup_indexes.py
DEFAULT_TOKENIZER_SCENARIO = "full"


# ── Main configuration ────────────────────────────────────────────────────────

@dataclass
class Config:
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    INDEX_DIR: Path = BASE_DIR / "data" / "indexes"
    REPORTS_DIR: Path = BASE_DIR / "reports"

    # Models — specter is tuned for scientific papers
    EMBEDDING_MODEL: str = "allenai/specter"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CLAUDE_MODEL: str = "claude-sonnet-4-6"

    # API
    ANTHROPIC_API_KEY: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    # Retrieval
    TOP_K_BM25: int = 100
    TOP_K_DENSE: int = 100
    TOP_K_HYBRID: int = 20
    TOP_K_RERANK: int = 5
    BM25_WEIGHT: float = 0.4
    DENSE_WEIGHT: float = 0.6
    RRF_K: int = 60

    # Tokenizer (BM25)
    tokenizer: TokenizerConfig = field(
        default_factory=lambda: TOKENIZER_SCENARIOS[DEFAULT_TOKENIZER_SCENARIO]
    )


    # Generation
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.1

    # Flask
    FLASK_DEBUG: bool = field(
        default_factory=lambda: os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    )
    FLASK_HOST: str = "0.0.0.0"
    FLASK_PORT: int = 5000
    SECRET_KEY: str = field(
        default_factory=lambda: os.environ.get(
            "SECRET_KEY", "dev-secret-key-change-in-prod"
        )
    )

config = Config()
