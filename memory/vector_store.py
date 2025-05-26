
### File: memory/vector_store.py
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_store"
))

task_collection = client.get_or_create_collection("tasks")

def add_to_vector_store(text, embedding, metadata):
    task_collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[metadata['id']]
    )

def search_similar_tasks(query_embedding, top_k=3):
    return task_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
