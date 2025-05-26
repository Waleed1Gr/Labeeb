# This file contains the LLM class that interacts with the OpenAI API to generate responses.

# import libraries

import openai
from .RAG.RAG_System import RAGSystem
import os
from functools import lru_cache
import hashlib

# create the LLM class

class LLM:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.retriever = RAGSystem(
            knowledge_base_dir="knowledge_base/documents"
        )
        # Update knowledge base on initialization
        self.retriever.update_knowledge_base()
        self.cache_size = 100  # Adjust based on memory
    
    @lru_cache(maxsize=100)
    def _cached_query(self, prompt_hash):
        # Implement caching logic
        # TODO: Implement caching logic here. For now, this is a placeholder.
        return None

    def generate_response(self, prompt, model="gpt-4-0125-preview", max_tokens=150):
# train it to speak saudi dialect
        saudi_system_prompt = """You are Labeeb, a friendly robot that speaks in Saudi Arabic dialect (not MSA).
        Always respond in Saudi dialect using common Saudi expressions and colloquialisms.
        Keep your tone warm and engaging, like a Saudi friend would speak."""
        
        context = self.retriever.query(prompt)
        
        messages = [
            {"role": "system", "content": saudi_system_prompt},
            {"role": "user", "content": f"Context: {' '.join(context)}\n\nQuestion: {prompt}"}
        ]
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
# write code to get the response from the LLM and send it to TTS
        return response.choices[0].message['content']

    def __del__(self):
        """Destructor to ensure cleanup when LLM instance is destroyed"""
        if hasattr(self, 'retriever'):
            self.retriever.cleanup()
