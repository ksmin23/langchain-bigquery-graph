"""Basic usage example for langchain-bigquery-graph.

This example demonstrates how to:
1. Create a BigQueryGraphStore and populate it with graph documents
2. Use BigQueryGraphVectorContextRetriever for vector-based graph search
3. Use BigQueryGraphTextToGQLRetriever for natural language to GQL

Prerequisites:
- A Google Cloud project with BigQuery API enabled
- A BigQuery dataset created beforehand
- Authentication configured (e.g., gcloud auth application-default login)
- pip install -r requirements.txt (or: pip install langchain-bigquery-graph[examples])
- Copy .env.example to .env and fill in your settings
"""

import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from langchain_bigquery_graph import (
    BigQueryGraphStore,
    BigQueryGraphTextToGQLRetriever,
    BigQueryGraphVectorContextRetriever,
    DistanceStrategy,
)

load_dotenv(find_dotenv())


def _env(name, *fallbacks, default=None):
    """Return the first non-None value from env vars, or default."""
    for key in (name, *fallbacks):
        val = os.environ.get(key)
        if val is not None:
            return val
    return default


def example_graph_store(project_id, location, dataset_id, graph_name, embedding_model_name):
    """Create a graph store and add documents."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    store = BigQueryGraphStore(
        project_id=project_id,
        dataset_id=dataset_id,
        graph_name=graph_name,
        location=location,
    )

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
    texts = ["Alice is a 30 year old person", "Bob is a 25 year old person", "Acme Corp is a company"]
    vectors = embeddings.embed_documents(texts)

    # Create graph documents with embeddings for vector search
    alice = Node(id="alice", type="Person", properties={"name": "Alice", "age": 30, "embedding": vectors[0]})
    bob = Node(id="bob", type="Person", properties={"name": "Bob", "age": 25, "embedding": vectors[1]})
    acme = Node(id="acme", type="Company", properties={"name": "Acme Corp", "embedding": vectors[2]})

    works_at = Relationship(source=alice, target=acme, type="WORKS_AT")
    knows = Relationship(source=alice, target=bob, type="KNOWS")

    doc = GraphDocument(
        nodes=[alice, bob, acme],
        relationships=[works_at, knows],
        source=Document(page_content="Alice works at Acme Corp and knows Bob."),
    )

    store.add_graph_documents([doc])
    print("Schema:", store.get_schema)

    # Query using GQL
    results = store.query(
        f"GRAPH `{dataset_id}`.`{graph_name}` MATCH (p:Person) RETURN p.name AS name"
    )
    print("Query results:", results)

    return store


def example_graph_vector_context_retriever(store, embedding_model_name):
    """Use the vector context retriever."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)

    retriever = BigQueryGraphVectorContextRetriever.from_params(
        embedding_service=embeddings,
        graph_store=store,
        label_expr="Person",
        embeddings_column="embedding",
        expand_by_hops=1,
        top_k=5,
        k=10,
    )

    docs = retriever.invoke("Who works at Acme?")
    for doc in docs:
        print(doc.page_content)


def example_text_to_gql_retriever(store, dataset_id, graph_name, llm_model_name, embedding_model_name):
    """Use the text-to-GQL retriever."""
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm = ChatGoogleGenerativeAI(model=llm_model_name)
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)

    retriever = BigQueryGraphTextToGQLRetriever.from_params(
        llm=llm,
        embedding_service=embeddings,
        graph_store=store,
        k=10,
    )

    retriever.add_example(
        question="Who works at Acme?",
        gql=f"GRAPH `{dataset_id}`.`{graph_name}` MATCH (p:Person)-[:WORKS_AT]->(c:Company {{name: 'Acme Corp'}}) RETURN p.name AS name",
    )

    docs = retriever.invoke("Find all people who work at Acme Corp")
    for doc in docs:
        print(doc.page_content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BigQuery Graph Store example")
    parser.add_argument("--project-id", default=_env("BIGQUERY_PROJECT", "GOOGLE_CLOUD_PROJECT"),
                        help="Google Cloud project ID (default: env BIGQUERY_PROJECT)")
    parser.add_argument("--location", default=_env("BIGQUERY_LOCATION", "GOOGLE_CLOUD_LOCATION"),
                        help="BigQuery location (default: env BIGQUERY_LOCATION)")
    parser.add_argument("--dataset-id", default=_env("BIGQUERY_DATASET"),
                        help="BigQuery dataset ID (default: env BIGQUERY_DATASET)")
    parser.add_argument("--graph-name", default=_env("BIGQUERY_GRAPH_NAME"),
                        help="Property graph name (default: env BIGQUERY_GRAPH_NAME)")
    parser.add_argument("--llm-model", default=_env("LLM_MODEL_NAME", default="gemini-2.5-flash"),
                        help="LLM model name (default: env LLM_MODEL_NAME or gemini-2.5-flash)")
    parser.add_argument("--embedding-model", default=_env("EMBEDDING_MODEL_NAME", default="gemini-embedding-001"),
                        help="Embedding model name (default: env EMBEDDING_MODEL_NAME or gemini-embedding-001)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove graph, tables, and all data after running")
    args = parser.parse_args()

    store = example_graph_store(args.project_id, args.location, args.dataset_id, args.graph_name, args.embedding_model)
    example_graph_vector_context_retriever(store, args.embedding_model)
    example_text_to_gql_retriever(store, args.dataset_id, args.graph_name, args.llm_model, args.embedding_model)

    if args.cleanup:
        store.cleanup()
