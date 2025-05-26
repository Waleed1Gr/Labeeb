from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import chromadb
from chromadb.config import Settings
import os

class RAGSystem:
    def __init__(self, knowledge_base_dir, persist_directory="./chroma_store"):
        self.knowledge_base_dir = knowledge_base_dir
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        self.collection = self.client.get_or_create_collection("knowledge_base")
        
        # Configure for lower memory usage
        self.chunk_size = 500  # Smaller chunks for RPi memory
        self.max_documents = 1000  # Limit total documents
        
    def load_documents(self):
        loader = DirectoryLoader(
            self.knowledge_base_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()[:self.max_documents]  # Limit documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=100  # Reduced overlap
        )
        return text_splitter.split_documents(documents)

    def update_knowledge_base(self):
        documents = self.load_documents()
        for i, doc in enumerate(documents):
            self.collection.add(
                documents=[doc.page_content],
                metadatas=[{"source": doc.metadata.get("source", "unknown")}],
                ids=[f"doc_{i}"]
            )

    def query(self, question, k=3):
        results = self.collection.query(
            query_texts=[question],
            n_results=k
        )
        return results["documents"][0] if results["documents"] else []

    def cleanup(self):
        """Persist changes to disk and cleanup resources"""
        self.client.persist()
