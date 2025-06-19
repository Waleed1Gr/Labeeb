import numpy as np
import faiss

def create_index(dimension: int = 384) -> faiss.IndexFlatL2:
    """
    Creates an in-memory FAISS index using L2 distance on vectors of `dimension`.
    """
    return faiss.IndexFlatL2(dimension)

def add_embedding(index: faiss.IndexFlatL2, embedding: list):
    """
    Adds a single embedding (list of floats) to the FAISS index.
    """
    index.add(np.array([embedding]))

def search_index(index: faiss.IndexFlatL2, query_emb: list, k: int = 5) -> list:
    """
    Searches the FAISS index for the top-k nearest neighbors to query_emb.
    Returns a list of k integer indices (or fewer if the index has fewer items).
    """
    D, I = index.search(np.array([query_emb]), k)
    return I[0].tolist()
