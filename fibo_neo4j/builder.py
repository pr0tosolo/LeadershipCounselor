"""Build a Neo4j-ready representation of the FIBO ontology."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import rdflib
from rdflib import Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD
from rdflib.util import guess_format

LOGGER = logging.getLogger(__name__)


def _format_literal(value: Literal) -> str:
    """Render a literal value with language/datatype hints."""
    text = str(value)
    if value.language:
        return f"{text} (@{value.language})"
    if value.datatype and value.datatype != XSD.string:
        return f"{text}^^{value.datatype}"  # include datatype when it is not xsd:string
    return text


def _split_namespace(uri: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a URI into namespace/local name, best effort."""
    try:
        namespace, local = rdflib.namespace.split_uri(URIRef(uri))
        return namespace, local
    except ValueError:
        if "#" in uri:
            namespace, local = uri.rsplit("#", 1)
            return namespace + "#", local or None
        if "/" in uri:
            namespace, local = uri.rsplit("/", 1)
            return namespace + "/", local or None
        return None, None


TYPE_LABEL_MAP: Dict[URIRef, str] = {
    OWL.Class: "Class",
    RDFS.Class: "Class",
    OWL.ObjectProperty: "ObjectProperty",
    OWL.DatatypeProperty: "DatatypeProperty",
    OWL.AnnotationProperty: "AnnotationProperty",
    OWL.NamedIndividual: "Individual",
    OWL.Ontology: "Ontology",
    RDF.Property: "Property",
    RDFS.Datatype: "Datatype",
    SKOS.Concept: "Concept",
    SKOS.ConceptScheme: "ConceptScheme",
    SKOS.Collection: "Collection",
}


LITERAL_PREDICATE_MAP: Dict[URIRef, str] = {
    RDFS.label: "names",
    SKOS.prefLabel: "pref_labels",
    SKOS.altLabel: "alt_labels",
    RDFS.comment: "comments",
    SKOS.definition: "definitions",
    SKOS.note: "notes",
    DCTERMS.title: "titles",
    DCTERMS.description: "descriptions",
    DCTERMS.identifier: "identifiers",
}


RELATIONSHIP_MAP: Dict[URIRef, str] = {
    RDFS.subClassOf: "SUBCLASS_OF",
    RDFS.subPropertyOf: "SUBPROPERTY_OF",
    RDFS.domain: "HAS_DOMAIN",
    RDFS.range: "HAS_RANGE",
    RDFS.isDefinedBy: "IS_DEFINED_BY",
    RDFS.seeAlso: "SEE_ALSO",
    OWL.equivalentClass: "EQUIVALENT_TO",
    OWL.equivalentProperty: "EQUIVALENT_PROPERTY",
    OWL.disjointWith: "DISJOINT_WITH",
    OWL.sameAs: "SAME_AS",
    OWL.inverseOf: "INVERSE_OF",
    OWL.imports: "IMPORTS",
    SKOS.broader: "BROADER_THAN",
    SKOS.narrower: "NARROWER_THAN",
    SKOS.related: "SKOS_RELATED",
    SKOS.inScheme: "IN_SCHEME",
}


RELATIONSHIP_FALLBACK_TYPE = "RELATED_TO"


@dataclass
class ResourceRecord:
    """Represents a node to be created in Neo4j."""

    uri: str
    labels: Set[str] = field(default_factory=lambda: {"Resource"})
    names: Set[str] = field(default_factory=set)
    pref_labels: Set[str] = field(default_factory=set)
    alt_labels: Set[str] = field(default_factory=set)
    comments: Set[str] = field(default_factory=set)
    definitions: Set[str] = field(default_factory=set)
    notes: Set[str] = field(default_factory=set)
    descriptions: Set[str] = field(default_factory=set)
    titles: Set[str] = field(default_factory=set)
    identifiers: Set[str] = field(default_factory=set)
    extra_annotations: Set[str] = field(default_factory=set)
    additional_types: Set[str] = field(default_factory=set)
    source_files: Set[str] = field(default_factory=set)
    namespace: Optional[str] = None
    local_name: Optional[str] = None
    deprecated: Optional[bool] = None

    def add_label(self, label: Optional[str]) -> None:
        if label:
            self.labels.add(label)

    def add_literal(self, predicate: URIRef, value: Literal) -> None:
        if predicate == OWL.deprecated:
            try:
                self.deprecated = bool(value.toPython())
            except Exception:  # pragma: no cover - rdflib handles datatypes
                self.deprecated = str(value).lower() == "true"
            return

        key = LITERAL_PREDICATE_MAP.get(predicate)
        formatted = _format_literal(value)
        if key:
            getattr(self, key).add(formatted)
        else:
            self.extra_annotations.add(f"{predicate}={formatted}")

    def add_type(self, uri: URIRef) -> None:
        label = TYPE_LABEL_MAP.get(uri)
        if label:
            self.labels.add(label)
        else:
            self.additional_types.add(str(uri))

    def to_serializable_dict(self) -> Dict[str, object]:
        return {
            "uri": self.uri,
            "namespace": self.namespace,
            "local_name": self.local_name,
            "names": sorted(self.names),
            "pref_labels": sorted(self.pref_labels),
            "alt_labels": sorted(self.alt_labels),
            "comments": sorted(self.comments),
            "definitions": sorted(self.definitions),
            "notes": sorted(self.notes),
            "descriptions": sorted(self.descriptions),
            "titles": sorted(self.titles),
            "identifiers": sorted(self.identifiers),
            "extra_annotations": sorted(self.extra_annotations),
            "additional_types": sorted(self.additional_types),
            "source_files": sorted(self.source_files),
            "deprecated": self.deprecated,
            "types": sorted(label for label in self.labels if label != "Resource"),
            "is_class": "Class" in self.labels,
            "is_object_property": "ObjectProperty" in self.labels,
            "is_datatype_property": "DatatypeProperty" in self.labels,
            "is_annotation_property": "AnnotationProperty" in self.labels,
            "is_individual": "Individual" in self.labels,
            "is_ontology": "Ontology" in self.labels,
            "is_concept": "Concept" in self.labels,
        }


@dataclass
class RelationshipRecord:
    """Represents a relationship to be created in Neo4j."""

    rel_type: str
    start_uri: str
    end_uri: str
    predicate: str
    source_files: Set[str] = field(default_factory=set)

    def key(self) -> Tuple[str, str, str, str]:
        return self.rel_type, self.start_uri, self.end_uri, self.predicate

    def to_serializable_dict(self) -> Dict[str, object]:
        return {
            "type": self.rel_type,
            "start_uri": self.start_uri,
            "end_uri": self.end_uri,
            "predicate": self.predicate,
            "source_files": sorted(self.source_files),
        }


@dataclass
class GraphData:
    """Container for node/relationship records."""

    nodes: List[ResourceRecord]
    relationships: List[RelationshipRecord]

    def node_dicts(self) -> List[Dict[str, object]]:
        return [node.to_serializable_dict() for node in self.nodes]

    def relationship_dicts(self) -> List[Dict[str, object]]:
        return [rel.to_serializable_dict() for rel in self.relationships]


class GraphBuilder:
    """Accumulates ontology content into Neo4j-friendly structures."""

    def __init__(
        self,
        *,
        file_extensions: Sequence[str] = (".ttl", ".rdf", ".owl"),
    ) -> None:
        self._file_extensions = tuple(file_extensions)
        self._nodes: Dict[str, ResourceRecord] = {}
        self._relationships: Dict[Tuple[str, str, str, str], RelationshipRecord] = {}
        self.triple_count = 0
        self.file_count = 0

    def build(self, path: Path, limit: Optional[int] = None) -> GraphData:
        """Parse ontology files within ``path`` and return :class:`GraphData`."""
        self.ingest_directory(path, limit=limit)
        return GraphData(list(self._nodes.values()), list(self._relationships.values()))

    def ingest_directory(self, path: Path, limit: Optional[int] = None) -> None:
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in self._file_extensions
        )
        if limit is not None:
            files = files[:limit]
        for file_path in files:
            self.ingest_file(file_path)

    def ingest_file(self, file_path: Path) -> None:
        fmt = guess_format(file_path.suffix)
        graph = rdflib.Graph()
        try:
            graph.parse(file_path, format=fmt)
        except Exception as exc:  # pragma: no cover - parse errors are rare but should be logged
            LOGGER.error("Failed to parse %s: %s", file_path, exc)
            return

        LOGGER.debug("Parsed %s with %d triples", file_path, len(graph))
        self.file_count += 1
        self.triple_count += len(graph)
        for subject, predicate, obj in graph:
            if isinstance(subject, URIRef):
                subject_record = self._get_or_create_resource(str(subject))
                subject_record.source_files.add(str(file_path))
                if predicate == RDF.type and isinstance(obj, URIRef):
                    subject_record.add_type(obj)
                    continue
                if isinstance(obj, Literal):
                    subject_record.add_literal(predicate, obj)
                    continue
            else:
                subject_record = None

            if isinstance(obj, URIRef):
                object_record = self._get_or_create_resource(str(obj))
                object_record.source_files.add(str(file_path))
                if isinstance(subject, URIRef):
                    self._add_relationship(predicate, subject_record, object_record, file_path)
            elif isinstance(obj, Literal) and predicate == RDF.type and subject_record is not None:
                subject_record.extra_annotations.add(f"{predicate}={_format_literal(obj)}")
            # Blank nodes and other value types are ignored for now.

    def _add_relationship(
        self,
        predicate: URIRef,
        subject_record: Optional[ResourceRecord],
        object_record: ResourceRecord,
        file_path: Path,
    ) -> None:
        if subject_record is None:
            return
        rel_type = RELATIONSHIP_MAP.get(predicate, RELATIONSHIP_FALLBACK_TYPE)
        record = RelationshipRecord(
            rel_type=rel_type,
            start_uri=subject_record.uri,
            end_uri=object_record.uri,
            predicate=str(predicate),
        )
        key = record.key()
        existing = self._relationships.get(key)
        if existing:
            existing.source_files.add(str(file_path))
        else:
            record.source_files.add(str(file_path))
            self._relationships[key] = record

    def _get_or_create_resource(self, uri: str) -> ResourceRecord:
        record = self._nodes.get(uri)
        if record is None:
            namespace, local_name = _split_namespace(uri)
            record = ResourceRecord(uri=uri, namespace=namespace, local_name=local_name)
            self._nodes[uri] = record
        return record

    def export_csv(self, output_dir: Path) -> Tuple[Path, Path]:
        """Export node and relationship data to CSV files compatible with Neo4j bulk import."""
        output_dir.mkdir(parents=True, exist_ok=True)
        graph_data = GraphData(list(self._nodes.values()), list(self._relationships.values()))
        nodes_path = output_dir / "nodes.csv"
        rels_path = output_dir / "relationships.csv"

        node_fieldnames = [
            "uri",
            "namespace",
            "local_name",
            "types",
            "names",
            "pref_labels",
            "alt_labels",
            "comments",
            "definitions",
            "notes",
            "descriptions",
            "titles",
            "identifiers",
            "extra_annotations",
            "additional_types",
            "source_files",
            "deprecated",
        ]
        rel_fieldnames = [
            "type",
            "start_uri",
            "end_uri",
            "predicate",
            "source_files",
        ]

        def _stringify(value: object) -> str:
            if isinstance(value, (list, tuple, set)):
                return json.dumps(list(value), ensure_ascii=False)
            if value is None:
                return ""
            if isinstance(value, bool):
                return "true" if value else "false"
            return str(value)

        with nodes_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=node_fieldnames)
            writer.writeheader()
            for node in graph_data.node_dicts():
                row = {key: _stringify(node.get(key)) for key in node_fieldnames}
                writer.writerow(row)

        with rels_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rel_fieldnames)
            writer.writeheader()
            for rel in graph_data.relationship_dicts():
                row = {key: _stringify(rel.get(key)) for key in rel_fieldnames}
                writer.writerow(row)

        return nodes_path, rels_path


def build_graph(
    path: Path,
    *,
    file_extensions: Sequence[str] = (".ttl", ".rdf", ".owl"),
    limit: Optional[int] = None,
) -> GraphData:
    """Convenience helper to build :class:`GraphData` from ``path``."""
    builder = GraphBuilder(file_extensions=file_extensions)
    return builder.build(path, limit=limit)
