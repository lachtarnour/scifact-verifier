"""
Text preprocessing utilities for the SciFact corpus.

The BM25 tokenizer is configurable through TokenizerConfig.
The same tokenizer configuration must be used at indexing and query time.
"""

import re
from typing import List, Tuple

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer, SnowballStemmer

from .load_scifact import CorpusType
from src.utils.config import TokenizerConfig, config


try:
    ENGLISH_STOP_WORDS: frozenset[str] = frozenset(
        nltk_stopwords.words("english")
    )
except LookupError as exc:
    raise RuntimeError(
        "NLTK stopwords are not available. "
        "Run: python -m nltk.downloader stopwords"
    ) from exc


SCIENTIFIC_STOP_WORDS: frozenset[str] = frozenset({
    "study", "studies",
    "result", "results",
    "conclusion", "conclusions",
    "method", "methods",
    "background", "objective", "purpose",
})

ALL_STOP_WORDS: frozenset[str] = ENGLISH_STOP_WORDS | SCIENTIFIC_STOP_WORDS

PORTER_STEMMER = PorterStemmer()
SNOWBALL_STEMMER = SnowballStemmer("english")


def build_doc_text(title: str, abstract: str) -> str:
    """Merge a paper title and abstract into a single searchable text."""
    title = title.strip()
    abstract = abstract.strip()

    if not title:
        return abstract

    separator = " " if title.endswith((".", "?", "!")) else ". "
    return f"{title}{separator}{abstract}"


def get_flat_corpus(corpus: CorpusType) -> Tuple[List[str], List[str]]:
    """
    Convert the corpus dictionary into parallel lists of document IDs and texts.

    The resulting texts are used for indexing, retrieval and evaluation.
    """
    doc_ids: List[str] = []
    doc_texts: List[str] = []

    for doc_id, document in corpus.items():
        title = document.get("title", "")
        abstract = document.get("text", "")

        doc_ids.append(doc_id)
        doc_texts.append(build_doc_text(title, abstract))

    return doc_ids, doc_texts




def tokenize_for_bm25(
    text: str,
    tokenizer_config: TokenizerConfig | None = None,
) -> List[str]:
    """
    Tokenize text for BM25 indexing and querying.

    The same TokenizerConfig must be used when building the BM25 index
    and when processing user queries.
    """
    cfg = tokenizer_config or config.tokenizer

    text = text.lower().strip()

    token_pattern = (
        r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b"
        if cfg.preserve_scientific_terms
        else r"\b[a-z0-9]+\b"
    )

    tokens = re.findall(token_pattern, text)

    if cfg.use_stop_words:
        tokens = [t for t in tokens if t not in ALL_STOP_WORDS]

    if not cfg.use_stemming:
        return [t for t in tokens if len(t) >= cfg.min_token_length]
    
    STEMMERS = {
        "porter": PORTER_STEMMER,
        "snowball": SNOWBALL_STEMMER,
        }

    stemmer = STEMMERS.get(cfg.stemmer)
    if stemmer is None:
        raise ValueError(
            f"Unsupported stemmer '{cfg.stemmer}'. "
            f"Expected one of: {list(STEMMERS.keys())}"
        )

    tokens = [
        t if (cfg.preserve_scientific_terms and "-" in t)
        else stemmer.stem(t)
        for t in tokens
    ]

    return [t for t in tokens if len(t) >= cfg.min_token_length]