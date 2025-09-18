from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence

import numpy as np

from fibo_neo4j.builder import GraphBuilder
from fibo_neo4j.glossary_linker import (
    CandidateMatch,
    ClassificationService,
    EmbeddingService,
    FiboOntologyIndex,
    GlossaryLinker,
    GlossaryTerm,
    SelectionResult,
)


DATA_DIR = Path(__file__).parent / "data"


class SimpleEmbeddingService(EmbeddingService):
    """Deterministic embedding service for tests using bag-of-words counts."""

    def __init__(self) -> None:
        vocabulary = ["agent", "customer", "account", "loan", "product", "service"]
        self._indices = {term: idx for idx, term in enumerate(vocabulary)}

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            vector = np.zeros(len(self._indices), dtype=np.float32)
            tokens = re.findall(r"[a-zA-Z]+", text.lower())
            for token in tokens:
                if token in self._indices:
                    vector[self._indices[token]] += 1.0
            embeddings.append(vector.tolist())
        return embeddings


class RuleBasedLLM(ClassificationService):
    """Simple heuristic for classification/selection during tests."""

    def classify_top_class(
        self, term: GlossaryTerm, candidate_classes: Sequence[str]
    ) -> str:
        text = f"{term.name} {term.description} {term.line_of_business}".lower()
        for label in candidate_classes:
            normalized = label.lower()
            if normalized and normalized in text:
                return label
        return candidate_classes[0]

    def select_best_candidate(
        self, term: GlossaryTerm, candidates: Sequence[CandidateMatch]
    ) -> SelectionResult:
        if not candidates:
            return SelectionResult(uri=None, rationale="no candidates")
        best = max(candidates, key=lambda item: item.similarity)
        return SelectionResult(uri=best.uri, rationale="highest similarity")


def build_index() -> FiboOntologyIndex:
    builder = GraphBuilder()
    graph_data = builder.build(DATA_DIR)
    service = SimpleEmbeddingService()
    index = FiboOntologyIndex(graph_data, service)
    index.build()
    return index


def test_glossary_linker_prefers_account() -> None:
    index = build_index()
    embedding_service = SimpleEmbeddingService()
    classifier = RuleBasedLLM()
    linker = GlossaryLinker(index, embedding_service, classifier)

    term = GlossaryTerm(
        name="Checking account",
        description="A customer deposit account for day to day banking",
    )
    result = linker.link_term(term, top_k=3)

    assert result.top_class == "account"
    assert result.top_class_uri == "http://example.com/Account"
    assert result.top_class_relationship == "HAS_DOMAIN"
    assert result.top_class_predicate == "http://www.w3.org/2000/01/rdf-schema#domain"
    candidate_uris = {candidate.uri for candidate in result.candidates}
    assert result.selection.uri in candidate_uris
    assert any(uri.endswith("Account") for uri in candidate_uris)


def test_glossary_linker_handles_unknown_class() -> None:
    index = build_index()
    embedding_service = SimpleEmbeddingService()
    classifier = RuleBasedLLM()
    linker = GlossaryLinker(index, embedding_service, classifier)

    term = GlossaryTerm(name="Generic resource", description="A concept without clear class")
    result = linker.link_term(term, top_k=2)

    assert result.top_class == "agent"
    assert result.top_class_uri == "http://example.com/Agent"
    assert len(result.candidates) >= 1
    assert result.selection.uri in {candidate.uri for candidate in result.candidates}


def test_top_level_classes_are_inferred_from_graph() -> None:
    index = build_index()
    labels = index.top_level_class_labels()
    assert labels == ["agent", "account", "loan", "product", "service"]
