"""Utilities for mapping glossary and dataset terms to FIBO concepts."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

from .builder import GraphBuilder, GraphData, ResourceRecord

LOGGER = logging.getLogger(__name__)

RDFS_DOMAIN_URI = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_DOMAIN_REL_TYPE = "HAS_DOMAIN"


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Embedding vector must be one-dimensional")
    norm = np.linalg.norm(arr)
    if not math.isfinite(norm) or norm == 0.0:
        return np.zeros_like(arr)
    return arr / norm


def _aggregate_text(record: ResourceRecord) -> Tuple[str, str, str]:
    """Create human-readable label/description/text payload for a node."""

    label_candidates: List[str] = []
    label_candidates.extend(record.pref_labels)
    label_candidates.extend(record.names)
    label_candidates.extend(record.titles)
    label_candidates.extend(record.alt_labels)
    if record.local_name:
        label_candidates.append(record.local_name)

    label = next((value for value in label_candidates if value), record.uri)

    description_parts: List[str] = []
    description_parts.extend(record.definitions)
    description_parts.extend(record.comments)
    description_parts.extend(record.descriptions)
    description = " \n".join(sorted(set(description_parts)))

    text_parts: List[str] = [label]
    text_parts.extend(record.pref_labels)
    text_parts.extend(record.names)
    text_parts.extend(record.alt_labels)
    text_parts.extend(record.titles)
    text_parts.extend(record.definitions)
    text_parts.extend(record.comments)
    text_parts.extend(record.descriptions)
    text_parts.extend(record.notes)
    text_parts.extend(record.identifiers)
    text = " \n".join(value for value in text_parts if value)

    return label, description, text


@dataclass
class GlossaryTerm:
    """Represents a glossary or dataset concept that needs alignment."""

    name: str
    description: str
    line_of_business: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_prompt(self) -> str:
        lines = [f"Term name: {self.name}"]
        if self.description:
            lines.append(f"Definition: {self.description}")
        if self.line_of_business:
            lines.append(f"Line of business: {self.line_of_business}")
        if self.metadata:
            # Only include scalar metadata to keep prompts concise
            extras = {
                key: value
                for key, value in self.metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            if extras:
                rendered = ", ".join(f"{k}={v}" for k, v in sorted(extras.items()))
                lines.append(f"Additional context: {rendered}")
        return "\n".join(lines)

    def embedding_text(self) -> str:
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        if self.line_of_business:
            parts.append(self.line_of_business)
        for value in self.metadata.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (int, float)):
                parts.append(str(value))
            elif isinstance(value, (list, tuple)):
                parts.extend(str(item) for item in value)
        return " \n".join(part for part in parts if part)


@dataclass
class CandidateMatch:
    uri: str
    label: str
    description: str
    similarity: float
    raw_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "label": self.label,
            "description": self.description,
            "similarity": self.similarity,
            "raw_text": self.raw_text,
        }


@dataclass
class SelectionResult:
    uri: Optional[str]
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "rationale": self.rationale,
        }


@dataclass
class LinkResult:
    term: GlossaryTerm
    top_class: str
    candidates: List[CandidateMatch]
    selection: SelectionResult
    top_class_uri: Optional[str] = None
    top_class_predicate: str = RDFS_DOMAIN_URI
    top_class_relationship: str = RDFS_DOMAIN_REL_TYPE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": {
                "name": self.term.name,
                "description": self.term.description,
                "line_of_business": self.term.line_of_business,
                "metadata": self.term.metadata,
            },
            "top_class": self.top_class,
            "top_class_uri": self.top_class_uri,
            "top_class_predicate": self.top_class_predicate,
            "top_class_relationship": self.top_class_relationship,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection": self.selection.to_dict(),
        }


@dataclass(frozen=True)
class _TopClassInfo:
    label: str
    normalized_label: str
    uri: Optional[str] = None
    predicate_uri: str = RDFS_DOMAIN_URI
    relationship_type: str = RDFS_DOMAIN_REL_TYPE

    def matches(self, value: str) -> bool:
        return _normalize_class_token(value) == self.normalized_label


class EmbeddingService(Protocol):
    """Protocol implemented by embedding providers."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return dense embeddings for each text."""


class ClassificationService(Protocol):
    """Protocol for high-level classification and selection."""

    def classify_top_class(
        self, term: GlossaryTerm, candidate_classes: Sequence[str]
    ) -> str:
        """Assign the glossary term to one of the provided classes."""

    def select_best_candidate(
        self, term: GlossaryTerm, candidates: Sequence[CandidateMatch]
    ) -> SelectionResult:
        """Choose the best ontology match from similarity-ranked candidates."""


class FiboOntologyIndex:
    """Vector index over FIBO resources for similarity search."""

    def __init__(self, graph_data: GraphData, embedding_service: EmbeddingService) -> None:
        self._graph_data = graph_data
        self._embedding_service = embedding_service
        self._nodes: List[ResourceRecord] = []
        self._labels: List[str] = []
        self._descriptions: List[str] = []
        self._texts: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._top_level_classes: Optional[Tuple[Tuple[str, str], ...]] = None

    def build(self) -> None:
        """Generate embeddings for every node in the graph."""

        self._nodes = []
        self._labels = []
        self._descriptions = []
        self._texts = []

        for record in self._graph_data.nodes:
            label, description, text = _aggregate_text(record)
            self._nodes.append(record)
            self._labels.append(label)
            self._descriptions.append(description)
            self._texts.append(text)

        embeddings = self._embedding_service.embed(self._texts)
        if len(embeddings) != len(self._texts):
            raise ValueError(
                "Embedding service returned mismatched number of vectors:"
                f" expected {len(self._texts)}, received {len(embeddings)}"
            )
        normalized = [_normalize_vector(vector) for vector in embeddings]
        self._embeddings = np.vstack(normalized) if normalized else np.zeros((0, 0), dtype=np.float32)

    def search(
        self,
        term_text: str,
        *,
        top_class: Optional[str] = None,
        top_k: int = 5,
        query_embedding: Optional[Sequence[float]] = None,
    ) -> List[CandidateMatch]:
        """Return the ``top_k`` most similar ontology resources."""

        if self._embeddings is None:
            raise RuntimeError("Index has not been built; call build() first")

        if query_embedding is None:
            query_vector = self._embedding_service.embed([term_text])[0]
        else:
            query_vector = query_embedding
        normalized_query = _normalize_vector(query_vector)

        if top_class:
            pattern = re.compile(re.escape(top_class), flags=re.IGNORECASE)
            indices = [idx for idx, text in enumerate(self._texts) if pattern.search(text)]
        else:
            indices = list(range(len(self._texts)))

        if not indices:
            indices = list(range(len(self._texts)))

        subset_embeddings = self._embeddings[indices]
        if subset_embeddings.size == 0:
            return []

        scores = subset_embeddings @ normalized_query
        order = np.argsort(scores)[::-1][:top_k]

        results: List[CandidateMatch] = []
        for local_rank in order:
            global_index = indices[int(local_rank)]
            record = self._nodes[global_index]
            label = self._labels[global_index]
            description = self._descriptions[global_index]
            raw_text = self._texts[global_index]
            score = float(scores[int(local_rank)])
            results.append(
                CandidateMatch(
                    uri=record.uri,
                    label=label,
                    description=description,
                    similarity=score,
                    raw_text=raw_text,
                )
            )
        return results

    # Ontology metadata ---------------------------------------------------
    def top_level_classes(self) -> List[Tuple[str, str]]:
        """Return cached list of ``(uri, label)`` pairs for root classes."""

        if self._top_level_classes is None:
            derived = _derive_top_level_classes(self._graph_data)
            self._top_level_classes = tuple(derived)
        return list(self._top_level_classes)

    def top_level_class_labels(self) -> List[str]:
        """Return normalized class labels for high-level classification."""

        return [label for _, label in self.top_level_classes()]


FALLBACK_TOP_CLASSES = (
    "agent",
    "customer",
    "account",
    "loan",
    "product",
    "service",
)


def _clean_class_label(label: str, record: ResourceRecord) -> str:
    """Normalize a class label into a lower-case token used for classification."""

    candidate = label.strip()
    if not candidate:
        candidate = record.local_name or record.uri
    if candidate.startswith("http://") or candidate.startswith("https://"):
        candidate = record.local_name or candidate.rsplit("#", 1)[-1] or candidate.rsplit("/", 1)[-1]
    if ":" in candidate and " " not in candidate:
        candidate = candidate.split(":")[-1]
    candidate = re.sub(r"\(@[a-zA-Z0-9_-]+\)", "", candidate)
    candidate = candidate.replace("_", " ").replace("-", " ")
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate.strip().lower()


def _derive_top_level_classes(graph_data: GraphData) -> List[Tuple[str, str]]:
    """Return pairs of (class URI, normalized label) for root classes in the ontology."""

    class_records: Dict[str, ResourceRecord] = {
        record.uri: record for record in graph_data.nodes if "Class" in record.labels
    }
    if not class_records:
        return []

    parent_map: Dict[str, Set[str]] = {uri: set() for uri in class_records}
    child_map: Dict[str, Set[str]] = {uri: set() for uri in class_records}
    for relationship in graph_data.relationships:
        if relationship.rel_type != "SUBCLASS_OF":
            continue
        child = relationship.start_uri
        parent = relationship.end_uri
        if child not in class_records:
            continue
        if parent in class_records:
            parent_map[child].add(parent)
            child_map.setdefault(parent, set()).add(child)

    top_candidates: List[Tuple[str, str, int]] = []
    for uri, record in class_records.items():
        parents = parent_map.get(uri, set())
        if parents:
            continue
        raw_label, _, _ = _aggregate_text(record)
        normalized_label = _clean_class_label(raw_label, record)
        if not normalized_label:
            continue
        child_count = len(child_map.get(uri, set()))
        top_candidates.append((uri, normalized_label, child_count))

    top_candidates.sort(key=lambda item: (-item[2], item[1], item[0]))
    seen: Set[str] = set()
    ordered: List[Tuple[str, str]] = []
    for uri, label, _ in top_candidates:
        if label in seen:
            continue
        seen.add(label)
        ordered.append((uri, label))
    return ordered


def _normalize_class_token(value: str) -> str:
    """Normalize arbitrary user-provided labels into canonical lower-case tokens."""

    token = value.strip().lower()
    token = token.replace("_", " ").replace("-", " ")
    token = re.sub(r"\s+", " ", token)
    return token


class GlossaryLinker:
    """Coordinates classification, similarity search and ontology alignment."""

    def __init__(
        self,
        index: FiboOntologyIndex,
        embedding_service: EmbeddingService,
        classifier: ClassificationService,
        *,
        top_classes: Optional[Sequence[str]] = None,
    ) -> None:
        self._index = index
        self._embedding_service = embedding_service
        self._classifier = classifier

        derived_infos: List[_TopClassInfo] = []
        seen_tokens: Set[str] = set()
        for uri, label in index.top_level_classes():
            token = _normalize_class_token(label)
            if not token or token in seen_tokens:
                continue
            seen_tokens.add(token)
            derived_infos.append(
                _TopClassInfo(label=label, normalized_label=token, uri=uri)
            )
        derived_map = {info.normalized_label: info for info in derived_infos}

        selected_infos: List[_TopClassInfo]
        if top_classes:
            selected_infos = []
            seen_tokens = set()
            for label in top_classes:
                if not label or not str(label).strip():
                    continue
                token = _normalize_class_token(str(label))
                if not token or token in seen_tokens:
                    continue
                seen_tokens.add(token)
                match = derived_map.get(token)
                if match:
                    selected_infos.append(match)
                else:
                    selected_infos.append(
                        _TopClassInfo(
                            label=str(label).strip(),
                            normalized_label=token,
                        )
                    )
        else:
            selected_infos = list(derived_infos)

        if not selected_infos:
            fallback_infos: List[_TopClassInfo] = []
            seen_tokens = set()
            for label in FALLBACK_TOP_CLASSES:
                token = _normalize_class_token(label)
                if not token or token in seen_tokens:
                    continue
                seen_tokens.add(token)
                match = derived_map.get(token)
                if match:
                    fallback_infos.append(match)
                else:
                    fallback_infos.append(
                        _TopClassInfo(label=label, normalized_label=token)
                    )
            selected_infos = fallback_infos

        if not selected_infos:
            raise ValueError("No top-level classes available for classification")

        self._top_class_infos: Tuple[_TopClassInfo, ...] = tuple(selected_infos)

    @property
    def top_classes(self) -> Tuple[str, ...]:
        """Classes used for the first-stage categorization."""

        return tuple(info.label for info in self._top_class_infos)

    def _resolve_top_class(self, label: str) -> _TopClassInfo:
        token = _normalize_class_token(label)
        if token:
            for info in self._top_class_infos:
                if info.normalized_label == token:
                    return info
        LOGGER.debug(
            "Classifier returned label '%s' which does not match known top classes;"
            " falling back to '%s'",
            label,
            self._top_class_infos[0].label,
        )
        return self._top_class_infos[0]

    def link_term(self, term: GlossaryTerm, *, top_k: int = 5) -> LinkResult:
        top_class_label = self._classifier.classify_top_class(term, self.top_classes)
        top_class_info = self._resolve_top_class(top_class_label)
        top_class = top_class_info.label
        text = term.embedding_text()
        embedding = self._embedding_service.embed([text])[0]
        candidates = self._index.search(
            text,
            top_class=top_class,
            top_k=top_k,
            query_embedding=embedding,
        )
        selection = self._classifier.select_best_candidate(term, candidates)
        return LinkResult(
            term=term,
            top_class=top_class,
            top_class_uri=top_class_info.uri,
            top_class_predicate=top_class_info.predicate_uri,
            top_class_relationship=top_class_info.relationship_type,
            candidates=list(candidates),
            selection=selection,
        )


def load_glossary_terms(path: Path) -> List[GlossaryTerm]:
    """Load glossary terms from a JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if "terms" in payload and isinstance(payload["terms"], list):
            items = payload["terms"]
        elif "items" in payload and isinstance(payload["items"], list):
            items = payload["items"]
        else:
            items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Glossary JSON must be a list or contain a 'terms' array")

    terms: List[GlossaryTerm] = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("Each glossary entry must be an object")
        name = raw.get("term") or raw.get("name") or raw.get("label")
        if not name:
            raise ValueError(f"Glossary entry missing term/name field: {raw}")
        description = (
            raw.get("description")
            or raw.get("definition")
            or raw.get("summary")
            or ""
        )
        line_of_business = raw.get("line_of_business") or raw.get("lineOfBusiness")
        metadata = {
            key: value
            for key, value in raw.items()
            if key
            not in {"term", "name", "label", "description", "definition", "summary", "line_of_business", "lineOfBusiness"}
        }
        terms.append(
            GlossaryTerm(
                name=str(name),
                description=str(description),
                line_of_business=str(line_of_business) if line_of_business is not None else None,
                metadata=metadata,
            )
        )
    return terms


def load_dataset_terms(path: Path) -> List[GlossaryTerm]:
    """Load dataset elements and convert them into :class:`GlossaryTerm` objects."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if "elements" in payload and isinstance(payload["elements"], list):
            items = payload["elements"]
        else:
            items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Dataset JSON must be a list or contain an 'elements' array")

    terms: List[GlossaryTerm] = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("Each dataset element must be an object")
        name = raw.get("name") or raw.get("element") or raw.get("column")
        if not name:
            raise ValueError(f"Dataset element missing name field: {raw}")
        description = (
            raw.get("description")
            or raw.get("logical_description")
            or raw.get("logicalDescription")
            or raw.get("definition")
            or ""
        )
        glossary_links = raw.get("glossary_terms") or raw.get("glossaryTerms") or []
        metadata = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "name",
                "element",
                "column",
                "description",
                "logical_description",
                "logicalDescription",
                "definition",
                "glossary_terms",
                "glossaryTerms",
            }
        }
        if glossary_links:
            metadata["glossary_terms"] = glossary_links
        terms.append(
            GlossaryTerm(
                name=str(name),
                description=str(description),
                metadata=metadata,
            )
        )
    return terms


def build_index_from_path(
    fibo_path: Path,
    embedding_service: EmbeddingService,
    *,
    limit_files: Optional[int] = None,
) -> FiboOntologyIndex:
    builder = GraphBuilder()
    graph_data = builder.build(fibo_path, limit=limit_files)
    LOGGER.info(
        "Built graph from %s files containing %s triples.", builder.file_count, builder.triple_count
    )
    index = FiboOntologyIndex(graph_data, embedding_service)
    index.build()
    LOGGER.info("Index contains %s ontology resources.", len(graph_data.nodes))
    return index


def link_terms(
    terms: Iterable[GlossaryTerm],
    linker: GlossaryLinker,
    *,
    top_k: int = 5,
) -> List[LinkResult]:
    results: List[LinkResult] = []
    for term in terms:
        try:
            result = linker.link_term(term, top_k=top_k)
            results.append(result)
        except Exception as exc:  # pragma: no cover - defensive logging for runtime failures
            LOGGER.error("Failed to link term '%s': %s", term.name, exc)
    return results


def results_to_json(results: Sequence[LinkResult]) -> List[Dict[str, Any]]:
    return [result.to_dict() for result in results]


def save_results(results: Sequence[LinkResult], output_path: Path) -> None:
    payload = results_to_json(results)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _parse_top_classes(value: Optional[str]) -> Optional[Sequence[str]]:
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(
        description="Link glossary or dataset terms to FIBO ontology concepts using Azure OpenAI."
    )
    parser.add_argument("fibo_path", type=Path, help="Path to the root of the FIBO repository")
    parser.add_argument(
        "--glossary-json", type=Path, help="Path to a glossary JSON file to align"
    )
    parser.add_argument(
        "--dataset-json", type=Path, help="Path to a dataset JSON file containing data elements"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write the alignment results as JSON"
    )
    parser.add_argument(
        "--limit-files", type=int, help="Parse only the first N ontology files (for testing)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of similarity candidates to evaluate"
    )
    parser.add_argument(
        "--top-classes",
        help="Comma separated override of the top-level classes used for initial classification",
    )

    args = parser.parse_args(argv)

    if not args.glossary_json and not args.dataset_json:
        parser.error("At least one of --glossary-json or --dataset-json must be provided")

    from .azure_client import AzureOpenAIService

    service = AzureOpenAIService()
    index = build_index_from_path(
        args.fibo_path, service, limit_files=args.limit_files
    )
    linker = GlossaryLinker(
        index,
        service,
        service,
        top_classes=_parse_top_classes(args.top_classes),
    )
    LOGGER.info("Using top-level classes: %s", ", ".join(linker.top_classes))

    terms: List[GlossaryTerm] = []
    if args.glossary_json:
        terms.extend(load_glossary_terms(args.glossary_json))
    if args.dataset_json:
        terms.extend(load_dataset_terms(args.dataset_json))

    results = link_terms(terms, linker, top_k=args.top_k)
    save_results(results, args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
