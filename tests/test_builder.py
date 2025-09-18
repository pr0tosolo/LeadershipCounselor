from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fibo_neo4j.builder import GraphBuilder


DATA_DIR = Path(__file__).parent / "data"


def test_builder_parses_sample_graph(tmp_path: Path) -> None:
    builder = GraphBuilder()
    graph_data = builder.build(DATA_DIR)

    assert builder.file_count == 1
    assert builder.triple_count > 0
    assert len(graph_data.nodes) >= 5

    nodes_by_uri = {node.uri: node for node in graph_data.nodes}
    account = nodes_by_uri["http://example.com/Account"]
    assert "Class" in account.labels
    assert "Account (@en)" in account.names
    assert "Represents an account." in account.comments

    customer = nodes_by_uri["http://example.com/Customer"]
    assert "Class" in customer.labels
    assert "Customer (@en)" in customer.pref_labels
    assert (
        "An individual or organization with an account."
        in customer.definitions
    )

    relationships = {(rel.rel_type, rel.start_uri, rel.end_uri) for rel in graph_data.relationships}
    assert (
        "SUBCLASS_OF",
        "http://example.com/Customer",
        "http://example.com/Agent",
    ) in relationships
    assert (
        "SUBCLASS_OF",
        "http://example.com/DepositoryAccount",
        "http://example.com/Account",
    ) in relationships
    assert (
        "HAS_DOMAIN",
        "http://example.com/hasAccount",
        "http://example.com/Customer",
    ) in relationships
    assert (
        "HAS_RANGE",
        "http://example.com/hasAccount",
        "http://example.com/Account",
    ) in relationships


def test_export_csv_creates_files(tmp_path: Path) -> None:
    builder = GraphBuilder()
    builder.build(DATA_DIR)
    nodes_path, rels_path = builder.export_csv(tmp_path)
    assert nodes_path.exists()
    assert rels_path.exists()
    assert nodes_path.stat().st_size > 0
    assert rels_path.stat().st_size > 0
