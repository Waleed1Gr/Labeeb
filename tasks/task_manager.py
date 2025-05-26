### File: tasks/task_manager.py
import uuid
from datetime import datetime
from models.embedder import get_embedding
from memory.vector_store import add_to_vector_store, search_similar_tasks

def add_task(text, time_str):
    task_id = str(uuid.uuid4())
    embedding = get_embedding(text)
    metadata = {
        "id": task_id,
        "text": text,
        "time": time_str,
        "created_at": datetime.now().isoformat()
    }
    add_to_vector_store(text, embedding, metadata)
    return metadata

def find_related_tasks(query_text):
    embedding = get_embedding(query_text)
    results = search_similar_tasks(embedding)
    tasks = [
        {
            "text": doc,
            "metadata": meta
        }
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ]
    return tasks
