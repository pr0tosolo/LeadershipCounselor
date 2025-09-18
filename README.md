# LeadershipCounselor
An Chatbot who helps you be more concise and drive better communication in your space

## FIBO → Neo4j loader

This repository now includes a reusable Python tool for converting the
[EDM Council FIBO ontology](https://github.com/edmcouncil/fibo) into a Neo4j
semantic graph. The tool parses the Turtle/RDF/OWL files in a local checkout of
the FIBO repository, extracts ontology concepts, properties, individuals and
relationships, and then either:

1. Loads the content directly into a running Neo4j instance using the official
   [`neo4j` Python driver](https://pypi.org/project/neo4j/), or
2. Exports CSV files (`nodes.csv` and `relationships.csv`) that are ready for
   Neo4j bulk import workflows.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

1. Clone or download the `edmcouncil/fibo` repository (or unpack a release
   archive) locally.
2. Run the CLI with the path to the FIBO sources:

```bash
python -m fibo_neo4j.cli /path/to/fibo \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j
```

You will be prompted for the Neo4j password if it is not supplied via the
`--neo4j-password` flag. To generate CSV output instead (or in addition), use
`--output-dir`:

```bash
python -m fibo_neo4j.cli /path/to/fibo --output-dir ./neo4j-export
```

Additional options:

- `--limit-files N` – parse only the first `N` ontology files (handy for
  smoke-testing the pipeline).
- `--batch-size N` – control the batch size used when writing to Neo4j.
- `--dry-run` – parse and report counts without touching Neo4j.
- `--no-constraints` – skip creation of the Neo4j uniqueness constraint on
  `Resource.uri`.

The loader captures structural relationships such as subclass hierarchies,
property domains/ranges, equivalence links, SKOS links, imports and more. Each
node stores labels, definitions, comments and provenance (source file list) to
aid downstream conversational AI workflows that rely on ontology-driven intent
clarification.
