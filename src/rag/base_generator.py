"""Abstract base class for LLM generators."""

from abc import ABC, abstractmethod
from collections.abc import Generator

from src.data.load_scifact import CorpusType


class BaseGenerator(ABC):
    """
    Every generator exposes:
      - generate()  → full response (dict)
      - stream()    → token-by-token generator (yields strings)
    """

    @abstractmethod
    def generate(
        self,
        claim: str,
        doc_ids: list[str],
        corpus: CorpusType,
    ) -> dict:
        """Return the full generation result as a dict."""
        ...

    @abstractmethod
    def stream(
        self,
        claim: str,
        doc_ids: list[str],
        corpus: CorpusType,
    ) -> Generator[str, None, None]:
        """Yield tokens one by one for streaming."""
        ...
