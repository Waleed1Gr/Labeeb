### File: models/embedder.py
from sentence_transformers import SentenceTransformer

# Load a small, fast model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text, convert_to_tensor=False).tolist()

