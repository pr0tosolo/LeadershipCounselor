"""Command line interface for building/loading FIBO into Neo4j."""
from __future__ import annotations

import argparse
import logging
from getpass import getpass
from pathlib import Path
from typing import Optional

from .builder import GraphBuilder
from .neo4j_loader import Neo4jLoader


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric, format="%(levelname)s %(name)s - %(message)s")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse FIBO ontology files and load them into Neo4j or export CSV files."
        )
    )
    parser.add_argument(
        "fibo_path",
        type=Path,
        help="Path to the root of the edmcouncil/fibo repository (or extracted release).",
    )
    parser.add_argument(
        "--neo4j-uri",
        dest="neo4j_uri",
        help="Bolt URI of the Neo4j database (e.g. bolt://localhost:7687).",
    )
    parser.add_argument(
        "--neo4j-user",
        dest="neo4j_user",
        default="neo4j",
        help="Neo4j user name (default: neo4j).",
    )
    parser.add_argument(
        "--neo4j-password",
        dest="neo4j_password",
        help="Neo4j password. If omitted and --neo4j-uri is provided, a prompt is shown.",
    )
    parser.add_argument(
        "--neo4j-database",
        dest="neo4j_database",
        help="Target Neo4j database name (optional, defaults to the server's default database).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Neo4j writes (default: 500).",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        help="Parse only the first N ontology files (useful for testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where CSV exports will be written (optional).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the ontologies but do not write to Neo4j (still writes CSV if requested).",
    )
    parser.add_argument(
        "--no-constraints",
        dest="create_constraints",
        action="store_false",
        help="Do not attempt to create Neo4j uniqueness constraints.",
    )
    parser.set_defaults(create_constraints=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    if not args.fibo_path.exists():
        raise FileNotFoundError(f"FIBO path not found: {args.fibo_path}")

    builder = GraphBuilder()
    graph_data = builder.build(args.fibo_path, limit=args.limit_files)
    logging.getLogger(__name__).info(
        "Parsed %s ontology files containing %d triples.",
        builder.file_count,
        builder.triple_count,
    )
    logging.getLogger(__name__).info(
        "Graph contains %d nodes and %d relationships.",
        len(graph_data.nodes),
        len(graph_data.relationships),
    )

    if args.output_dir is not None:
        nodes_path, rels_path = builder.export_csv(args.output_dir)
        logging.getLogger(__name__).info(
            "Wrote CSV exports: nodes=%s, relationships=%s", nodes_path, rels_path
        )

    if args.neo4j_uri:
        if args.dry_run:
            logging.getLogger(__name__).info(
                "Dry run requested; skipping Neo4j writes while keeping connection parameters."
            )
            return
        password = args.neo4j_password
        if password is None:
            prompt = f"Neo4j password for user {args.neo4j_user}: "
            password = getpass(prompt)
        loader = Neo4jLoader(
            args.neo4j_uri,
            user=args.neo4j_user,
            password=password,
            database=args.neo4j_database,
            batch_size=args.batch_size,
        )
        try:
            loader.load(
                graph_data,
                create_constraints=args.create_constraints,
                dry_run=False,
            )
        finally:
            loader.close()
    else:
        logging.getLogger(__name__).info("Neo4j loading skipped (no URI provided).")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
