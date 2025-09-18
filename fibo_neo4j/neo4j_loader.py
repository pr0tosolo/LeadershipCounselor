"""Utilities for pushing ontology content into Neo4j."""
from __future__ import annotations

import logging
from itertools import islice
from typing import Iterable, Iterator, List, Optional, Sequence

from neo4j import GraphDatabase, Session

from .builder import GraphData

LOGGER = logging.getLogger(__name__)


def _chunked(iterable: Sequence[dict] | Iterable[dict], size: int) -> Iterator[List[dict]]:
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk


class Neo4jLoader:
    """Pushes :class:`GraphData` content into a Neo4j database."""

    def __init__(
        self,
        uri: str,
        *,
        user: str,
        password: str,
        database: Optional[str] = None,
        batch_size: int = 500,
    ) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._batch_size = batch_size

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jLoader":  # pragma: no cover - trivial context mgmt
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial context mgmt
        self.close()

    def load(
        self,
        graph_data: GraphData,
        *,
        create_constraints: bool = True,
        dry_run: bool = False,
    ) -> None:
        node_rows = graph_data.node_dicts()
        rel_rows = graph_data.relationship_dicts()
        LOGGER.info(
            "Graph contains %d nodes, %d relationships", len(node_rows), len(rel_rows)
        )
        if dry_run:
            return

        with self._driver.session(database=self._database) as session:
            if create_constraints:
                self._ensure_constraints(session)
            self._load_nodes(session, node_rows)
            self._load_relationships(session, rel_rows)

    def _ensure_constraints(self, session: Session) -> None:
        session.run(
            "CREATE CONSTRAINT fibo_resource_uri IF NOT EXISTS "
            "FOR (n:Resource) REQUIRE n.uri IS UNIQUE"
        )

    def _load_nodes(self, session: Session, nodes: List[dict]) -> None:
        query = (
            "UNWIND $rows AS row\n"
            "MERGE (n:Resource {uri: row.uri})\n"
            "SET n.namespace = row.namespace,\n"
            "    n.local_name = row.local_name,\n"
            "    n.names = row.names,\n"
            "    n.pref_labels = row.pref_labels,\n"
            "    n.alt_labels = row.alt_labels,\n"
            "    n.comments = row.comments,\n"
            "    n.definitions = row.definitions,\n"
            "    n.notes = row.notes,\n"
            "    n.descriptions = row.descriptions,\n"
            "    n.titles = row.titles,\n"
            "    n.identifiers = row.identifiers,\n"
            "    n.extra_annotations = row.extra_annotations,\n"
            "    n.additional_types = row.additional_types,\n"
            "    n.source_files = row.source_files,\n"
            "    n.types = row.types,\n"
            "    n.deprecated = row.deprecated\n"
            "FOREACH (_ IN CASE WHEN row.is_class THEN [1] ELSE [] END | SET n:Class)\n"
            "FOREACH (_ IN CASE WHEN row.is_object_property THEN [1] ELSE [] END | SET n:ObjectProperty)\n"
            "FOREACH (_ IN CASE WHEN row.is_datatype_property THEN [1] ELSE [] END | SET n:DatatypeProperty)\n"
            "FOREACH (_ IN CASE WHEN row.is_annotation_property THEN [1] ELSE [] END | SET n:AnnotationProperty)\n"
            "FOREACH (_ IN CASE WHEN row.is_individual THEN [1] ELSE [] END | SET n:Individual)\n"
            "FOREACH (_ IN CASE WHEN row.is_ontology THEN [1] ELSE [] END | SET n:Ontology)\n"
            "FOREACH (_ IN CASE WHEN row.is_concept THEN [1] ELSE [] END | SET n:Concept)"
        )
        for chunk in _chunked(nodes, self._batch_size):
            session.run(query, rows=chunk)

    def _load_relationships(self, session: Session, relationships: List[dict]) -> None:
        if not relationships:
            return
        rels_by_type: dict[str, List[dict]] = {}
        for rel in relationships:
            rels_by_type.setdefault(rel["type"], []).append(rel)

        for rel_type, rows in rels_by_type.items():
            query = (
                "UNWIND $rows AS row\n"
                "MATCH (start:Resource {uri: row.start_uri})\n"
                "MATCH (end:Resource {uri: row.end_uri})\n"
                f"MERGE (start)-[rel:{rel_type} {{predicate: row.predicate}}]->(end)\n"
                "SET rel.source_files = row.source_files"
            )
            for chunk in _chunked(rows, self._batch_size):
                session.run(query, rows=chunk)
