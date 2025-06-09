import os
from openai import OpenAI

# Initialize OpenAI client (ensure OPENAI_API_KEY is set)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def get_embedding(text: str) -> list:
    """
    Returns a 384-dimensional embedding for the given text
    using the `text-embedding-ada-002` model.
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding API error: {e}")
        return []
