"""
Abstract base class for all retrievers.

Every retriever in the pipeline exposes at least a `retrieve()` method so that
`evaluate_retrieval()` can call any of them uniformly.

`build()`, `load()` and `_save()` are optional — composite retrievers like
HybridRetriever and RerankedRetriever delegate indexing to their sub-retrievers
and therefore don't need their own index lifecycle.
"""

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Minimal interface shared by every retriever."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> dict[str, float]:
        """Return ``{doc_id: score}`` for the *top_k* most relevant documents."""
        ...

    # ── Optional lifecycle hooks (index-based retrievers override these) ──

    def build(self, corpus, **kwargs) -> None:  # noqa: ARG002
        """Build the underlying index from *corpus*.  Override if needed."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support build()."
        )

    def load(self) -> bool:
        """Load a previously saved index.  Return True on success."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support load()."
        )

    def _save(self) -> None:
        """Persist the index to disk.  Override if needed."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support _save()."
        )
