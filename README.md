# langchain-bigquery-graph

LangChain integration for [BigQuery Graph](https://cloud.google.com/bigquery/docs/graph-overview). Provides a `GraphStore` implementation and two retriever classes for building Graph RAG applications with BigQuery.

## Features

- **BigQueryGraphStore** -- `GraphStore` interface for BigQuery Graphs (schema management, GQL queries, graph document ingestion)
- **BigQueryGraphVectorContextRetriever** -- Vector similarity search on graph nodes with optional multi-hop neighborhood expansion
- **BigQueryGraphTextToGQLRetriever** -- LLM-powered natural language to GQL translation with optional few-shot examples

## Installation

```bash
pip install langchain-bigquery-graph
```

For development:

```bash
pip install langchain-bigquery-graph[dev]
```

## Prerequisites

- Python 3.10+
- A Google Cloud project with BigQuery API enabled
- Authentication configured:
  ```bash
  gcloud auth application-default login
  ```
- Set the following environment variable to use Vertex AI with ADC (no API key required):
  ```bash
  export GOOGLE_GENAI_USE_VERTEXAI=true
  ```
- A BigQuery dataset must be created before using `BigQueryGraphStore`. Tables and property graphs are created automatically, but the dataset is not.
  ```bash
  bq mk --dataset --location=us-central1 YOUR_PROJECT_ID:YOUR_DATASET
  ```

## Quick Start

### 1. Create a Graph Store and Add Data

```python
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_bigquery_graph import BigQueryGraphStore

store = BigQueryGraphStore(
    project_id="my-project",
    dataset_id="my_dataset",
    graph_name="knowledge_graph",
    location="us-central1",
)

# Define nodes and relationships
alice = Node(id="alice", type="Person", properties={"name": "Alice", "age": 30})
bob = Node(id="bob", type="Person", properties={"name": "Bob", "age": 25})
acme = Node(id="acme", type="Company", properties={"name": "Acme Corp"})

works_at = Relationship(source=alice, target=acme, type="WORKS_AT")
knows = Relationship(source=alice, target=bob, type="KNOWS")

doc = GraphDocument(
    nodes=[alice, bob, acme],
    relationships=[works_at, knows],
    source=Document(page_content="Alice works at Acme Corp and knows Bob."),
)

# This creates tables, the property graph, and inserts data
store.add_graph_documents([doc])

# Query with GQL
results = store.query(
    "GRAPH `my_dataset`.`knowledge_graph` MATCH (p:Person) RETURN p.name AS name"
)
print(results)
# [{'name': 'Alice'}, {'name': 'Bob'}]
```

### 2. Vector Context Retriever

Search graph nodes by vector similarity and optionally expand results by traversing the graph neighborhood. Vector search runs as SQL on the base table (since BigQuery property graphs don't support ARRAY properties), while graph traversal uses GQL.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_bigquery_graph import BigQueryGraphVectorContextRetriever

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Return specific properties from matching nodes
retriever = BigQueryGraphVectorContextRetriever(
    graph_store=store,
    embedding_service=embeddings,
    label_expr="Person",
    embeddings_column="embedding",
    return_properties_list=["name", "age"],
    top_k=5,
    k=10,
)
docs = retriever.invoke("Who works at Acme?")
```

With multi-hop expansion:

```python
# Expand results by traversing 2 hops from matching nodes
retriever = BigQueryGraphVectorContextRetriever(
    graph_store=store,
    embedding_service=embeddings,
    label_expr="Person",
    expand_by_hops=2,
    top_k=5,
)
docs = retriever.invoke("Tell me about Alice")
```

You can also use the factory method:

```python
retriever = BigQueryGraphVectorContextRetriever.from_params(
    embedding_service=embeddings,
    graph_store=store,
    expand_by_hops=1,
)
```

### 3. Text-to-GQL Retriever

Translate natural language questions into GQL queries using an LLM, execute them, and return the results.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_bigquery_graph import BigQueryGraphTextToGQLRetriever

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

retriever = BigQueryGraphTextToGQLRetriever.from_params(
    llm=llm,
    graph_store=store,
    k=10,
)

docs = retriever.invoke("Find all people who work at Acme Corp")
```

With few-shot examples for better GQL generation:

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

retriever = BigQueryGraphTextToGQLRetriever.from_params(
    llm=llm,
    embedding_service=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    graph_store=store,
)

retriever.add_example(
    question="Who works at Acme?",
    gql="GRAPH `my_dataset`.`knowledge_graph` MATCH (p:Person)-[:WORKS_AT]->(c:Company {name: 'Acme Corp'}) RETURN p.name AS name",
)

docs = retriever.invoke("Which people are employed by Acme Corp?")
```

## API Reference

### BigQueryGraphStore

| Parameter | Type | Default | Description |
|---|---|---|---|
| `project_id` | `str` | required | Google Cloud project ID |
| `dataset_id` | `str` | required | BigQuery dataset ID |
| `graph_name` | `str` | required | Property graph name |
| `client` | `bigquery.Client` | `None` | Optional pre-configured client |
| `location` | `str` | `None` | BigQuery location (e.g., `us-central1`). Ignored if `client` is provided |
| `use_flexible_schema` | `bool` | `False` | Use JSON-based flexible schema |
| `static_node_properties` | `List[str]` | `[]` | Properties stored as static columns in flexible schema |
| `static_edge_properties` | `List[str]` | `[]` | Properties stored as static columns in flexible schema |

**Methods:**

| Method | Description |
|---|---|
| `query(query, params)` | Execute a GQL query and return results |
| `get_schema` | Property graph schema as JSON string |
| `get_structured_schema` | Schema as a Python dictionary |
| `add_graph_documents(docs)` | Create tables, graph DDL, and insert data |
| `refresh_schema()` | Reload schema from `INFORMATION_SCHEMA` |
| `cleanup()` | Drop property graph and all associated tables |

### BigQueryGraphVectorContextRetriever

| Parameter | Type | Default | Description |
|---|---|---|---|
| `graph_store` | `BigQueryGraphStore` | required | The graph store to search |
| `embedding_service` | `Embeddings` | required | Embedding model for vectorizing queries |
| `label_expr` | `str` | `"%"` | Label expression to filter nodes |
| `return_properties_list` | `List[str]` | `[]` | Specific properties to return (mutually exclusive with `expand_by_hops`) |
| `embeddings_column` | `str` | `"embedding"` | Column name storing node embeddings |
| `distance_strategy` | `DistanceStrategy` | `COSINE` | `COSINE` or `EUCLIDEAN` |
| `top_k` | `int` | `3` | Number of vector similarity matches |
| `expand_by_hops` | `int` | `-1` | Hops to traverse for neighborhood expansion (mutually exclusive with `return_properties_list`) |
| `k` | `int` | `10` | Max number of graph results to return |

> **Note:** Exactly one of `return_properties_list` or `expand_by_hops` must be provided. With `return_properties_list`, results come from a direct SQL query on the base table. With `expand_by_hops`, vector search finds matching node IDs via SQL, then GQL traverses the graph neighborhood.

### BigQueryGraphTextToGQLRetriever

| Parameter | Type | Default | Description |
|---|---|---|---|
| `graph_store` | `BigQueryGraphStore` | required | The graph store to query |
| `llm` | `BaseLanguageModel` | required | LLM for GQL generation |
| `k` | `int` | `10` | Max number of results to return |
| `selector` | `SemanticSimilarityExampleSelector` | `None` | Few-shot example selector (auto-created via `from_params`) |

### DistanceStrategy

```python
from langchain_bigquery_graph import DistanceStrategy

DistanceStrategy.COSINE      # COSINE_DISTANCE
DistanceStrategy.EUCLIDEAN   # EUCLIDEAN_DISTANCE
```

## Examples

```bash
cd examples
cp .env.example .env
# Edit .env with your project settings

pip install -r requirements.txt
# or: pip install -e "..[examples]"

python basic_usage.py
python basic_usage.py --cleanup  # remove graph and tables after running
```

## Architecture

```
langchain-bigquery-graph/
├── src/langchain_bigquery_graph/
│   ├── __init__.py            # Public exports
│   ├── graph_store.py         # BigQueryGraphStore, BigQueryGraphSchema, ElementSchema
│   ├── graph_retriever.py     # BigQueryGraphVectorContextRetriever,
│   │                          # BigQueryGraphTextToGQLRetriever, DistanceStrategy
│   ├── graph_utils.py         # GQL syntax fixing and extraction
│   └── prompts.py             # GQL generation prompt templates
├── tests/
│   ├── test_graph_store.py
│   └── test_graph_retriever.py
├── examples/                  # Jupyter notebooks and Python example scripts
└── pyproject.toml
```

### Design Decisions

| Decision | Approach | Rationale |
|---|---|---|
| Upsert | `MERGE INTO ... WHEN MATCHED / NOT MATCHED` | BigQuery lacks `INSERT OR UPDATE` |
| Primary Key | `NOT ENFORCED` | BigQuery advisory-only PKs |
| Graph DDL | `CREATE OR REPLACE PROPERTY GRAPH` | Idempotent creation; supports schema evolution on existing graphs |
| Vector Search | SQL on base table with `COSINE_DISTANCE` / `EUCLIDEAN_DISTANCE` | Property graphs don't support ARRAY properties; vector search uses SQL, graph traversal uses GQL |
| JSON conversion | `TO_JSON` | BigQuery equivalent of Spanner's `SAFE_TO_JSON` |
| Schema evolution | `ALTER TABLE ADD COLUMN IF NOT EXISTS` | Idempotent; safely adds new properties even if table already exists |
| ARRAY properties | Stored in table, excluded from `LABEL PROPERTIES` | BigQuery property graphs don't support ARRAY types; data is stored but not exposed as graph properties |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All tests use mocked BigQuery clients and do not require a real BigQuery connection.

## Related Projects

- [langchain-google-spanner-python](https://github.com/googleapis/langchain-google-spanner-python) -- Spanner Graph integration that this project is based on
- [LightRAG](https://github.com/HKUDS/LightRAG) -- Simple and Fast Retrieval-Augmented Generation that incorporates graph structures into text indexing and retrieval processes.
  - [lightrag-bigquery](https://github.com/ksmin23/lightrag-bigquery) -- BigQuery storage backend for [LightRAG](https://github.com/HKUDS/LightRAG)
- [PathRAG](https://github.com/ksmin23/PathRAG) -- A Path-based Retrieval-Augmented Generation (PathRAG) library. Contributed the Google Cloud Spanner storage backend (Graph, Vector, KV) and LiteLLM/Gemini model support to the original framework.
  - [pathrag-bigquery](https://github.com/ksmin23/pathrag-bigquery) -- A Google Cloud BigQuery storage plugin for [PathRAG](https://github.com/ksmin23/PathRAG). It provides KV Vector, and Graph storage classes as an external plugin — no modifications to PathRAG source code required.


## License

MIT License. See [LICENSE](LICENSE) for details.
