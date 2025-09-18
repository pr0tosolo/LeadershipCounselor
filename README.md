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

## Glossary and dataset alignment via Azure OpenAI

The project now includes utilities for linking large, flat business glossaries
and dataset dictionaries to the FIBO ontology using your Azure OpenAI
endpoint. The workflow automatically derives broad top-level FIBO classes (e.g.
`agent`, `account`, `loan`, `product`, `service`) from the ontology, classifies
each glossary term into those buckets, narrows the search space via vector
similarity over the FIBO graph, and finally prompts the LLM to pick the best
ontology match. When the classifier selects a top class, the linker records the
corresponding class URI and attaches an `rdfs:domain` (`HAS_DOMAIN`) link in the
JSON output so downstream tooling can assert the relationship between the term
and the inferred high-level ontology class.

### Configuration

Provide your Azure OpenAI credentials via environment variables before running
the linker:

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o-mini"
```

### Running the linker

Glossary JSON files should contain a list of objects with at least `term`/`name`
and `description` fields, plus optional `line_of_business` metadata. Dataset
JSON files follow a similar structure using `name` and `description` fields and
may include `glossary_terms` arrays for extra context.

```bash
python -m fibo_neo4j.glossary_linker \
  /path/to/fibo \
  --glossary-json glossary.json \
  --dataset-json dataset.json \
  --output alignment-results.json \
  --top-k 5
```

The command builds embeddings for the FIBO ontology, classifies each input
entry, retrieves the top similarity matches, and records the LLM-chosen URI
alongside the candidates and rationale. Results are written as JSON for further
review or downstream automation.
