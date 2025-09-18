"""Utilities for converting FIBO ontologies into Neo4j graphs."""

__all__ = [
    "build_graph",
]

from .builder import build_graph  # noqa: E402  (import after definition for __all__)
