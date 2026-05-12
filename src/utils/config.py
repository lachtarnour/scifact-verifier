import os
from pathlib import Path
from dataclasses import dataclass, field 

BASE_DIR = Path(__file__).resolve().parent.parent.parent

@dataclass
class Config:
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    INDEX_DIR: Path = BASE_DIR / "data" / "index"   
    REPORTS_DIR: Path = BASE_DIR / "reports"    


from dataclasses import dataclass
from typing import Literal


@dataclass
class TokenizerConfig:
    """
    Configuration used to tokenize text before BM25 indexing and querying.

    The same configuration must be used at indexing time and query time.
    If one of these parameters changes, the BM25 index should be rebuilt.
    """

    use_stemming: bool = True
    # Apply stemming to reduce inflected forms to a common root.

    stemmer: Literal["porter", "snowball"] = "porter"
    # "porter"  : fast and aggressive.
    # "snowball": slightly more modern and usually less aggressive.

    use_stop_words: bool = True
    # Remove common English stop words and a small list of scientific filler words.

    preserve_scientific_terms: bool = True
    # Keep hyphenated scientific terms intact, such as "IL-6", "COVID-19",
    # or "TNF-alpha".

    min_token_length: int = 2
    # Remove very short tokens such as isolated letters or digits.

    def __post_init__(self) -> None:
        if self.stemmer not in {"porter", "snowball"}:
            raise ValueError(
                "stemmer must be either 'porter' or 'snowball'."
            )

        if self.min_token_length < 1:
            raise ValueError(
                "min_token_length must be at least 1."
            )