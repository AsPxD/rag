"""
FAISS-based vector store for fast document retrieval
Optimized for deployment on Render/Vercel with minimal dependencies
"""
import os
import pickle
import time
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FAISSStore:
    """FAISS vector store with document retrieval using existing LangChain setup"""
    
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = index_path
        self.vectorstore = None
        # Configure Google API
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        else:
            print("Warning: GOOGLE_API_KEY not found. Embeddings may not work.")
            self.embeddings = None
        self.top_k = 4
        
    def load_index(self):
        """Load FAISS index using LangChain"""
        print("Loading FAISS index...")
        start_time = time.time()
        
        try:
            # Load existing FAISS vectorstore
            if os.path.exists(self.index_path):
                self.vectorstore = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✓ FAISS vectorstore loaded successfully")
            else:
                raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
            
            load_time = time.time() - start_time
            print(f"✓ FAISS store ready in {load_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Error loading FAISS store: {e}")
            raise
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query using LangChain FAISS"""
        if self.vectorstore is None:
            raise RuntimeError("FAISS store not loaded. Call load_index() first.")
        
        try:
            # Use LangChain's similarity search
            docs = self.vectorstore.similarity_search(query, k=self.top_k)
            
            # Extract page content from documents
            retrieved_docs = [doc.page_content for doc in docs]
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    async def async_retrieve(self, query: str) -> List[str]:
        """Async wrapper for retrieve method"""
        import asyncio
        return await asyncio.to_thread(self.retrieve, query)

# Global store instance
store = FAISSStore()

def get_store() -> FAISSStore:
    """Get the global FAISS store instance"""
    return store

def retrieve_documents(query: str) -> List[str]:
    """Convenience function for document retrieval"""
    return store.retrieve(query)

async def async_retrieve_documents(query: str) -> List[str]:
    """Async convenience function for document retrieval"""
    return await store.async_retrieve(query)
