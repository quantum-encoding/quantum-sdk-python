"""RAG module — re-exports RAG types for convenience."""

from .types import (
    RAGCorpus,
    RAGResult,
    RAGSearchRequest,
    RAGSearchResponse,
    SurrealRAGResult,
    SurrealRAGSearchRequest,
    SurrealRAGSearchResponse,
)

__all__ = [
    "RAGCorpus",
    "RAGResult",
    "RAGSearchRequest",
    "RAGSearchResponse",
    "SurrealRAGResult",
    "SurrealRAGSearchRequest",
    "SurrealRAGSearchResponse",
]
