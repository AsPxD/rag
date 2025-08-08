"""
Optimized FastAPI RAG system with FAISS and Gemini 2.5 Pro
Designed for fast deployment on Render/Vercel with streaming responses
"""
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm import get_llm
from store import get_store
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG Chatbot API",
    version="2.0.0",
    description="Fast RAG system with FAISS and Gemini 2.5 Pro"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
llm = get_llm()
store = get_store()

# Request models
class Query(BaseModel):
    query: str

class HealthStatus(BaseModel):
    status: str = "healthy"
    timestamp: str = ""
    faiss_loaded: bool = False
    total_documents: int = 0

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time*1000:.1f}ms"
    return response

# Medical query keywords for fast filtering
MEDICAL_KEYWORDS = {
    'medical', 'health', 'disease', 'symptom', 'treatment', 'medicine', 'doctor',
    'patient', 'diagnosis', 'therapy', 'medication', 'hospital', 'clinic',
    'seizure', 'epilepsy', 'brain', 'neurological', 'pain', 'fever', 'infection',
    'surgery', 'prescription', 'drug', 'pharmaceutical', 'healthcare', 'wellness'
}

def is_medical_query(query: str) -> bool:
    """Fast medical query detection"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in MEDICAL_KEYWORDS)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("🚀 Starting Medical RAG API...")
    
    try:
        # Load FAISS index
        store.load_index()
        print("✅ Application ready!")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    try:
        faiss_loaded = store.vectorstore is not None
        total_docs = 0
        if hasattr(store, 'vectorstore') and store.vectorstore is not None:
            # Try to get document count safely
            try:
                total_docs = len(store.vectorstore.docstore._dict) if hasattr(store.vectorstore, 'docstore') else 0
            except:
                total_docs = 0
        
        return HealthStatus(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            faiss_loaded=faiss_loaded,
            total_documents=total_docs
        )
    except Exception as e:
        return HealthStatus(
            status="error",
            timestamp=datetime.now().isoformat(),
            faiss_loaded=False,
            total_documents=0
        )

@app.post("/rag")
async def rag_endpoint(query: Query):
    """Main RAG endpoint with streaming response"""
    start_time = time.time()
    
    try:
        # Validate query
        if not query.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if it's a medical query
        if not is_medical_query(query.query):
            return {
                "answer": "I'm a medical chatbot focused on health topics. Please ask a medical question.",
                "sources": "N/A",
                "processing_time": round(time.time() - start_time, 3),
                "timestamp": datetime.now().isoformat()
            }
        
        # Retrieve relevant documents
        docs = await store.async_retrieve(query.query)
        
        if not docs:
            return {
                "answer": "I couldn't find relevant information in my medical knowledge base. Please try rephrasing your question.",
                "sources": "Medical Knowledge Base",
                "processing_time": round(time.time() - start_time, 3),
                "timestamp": datetime.now().isoformat()
            }
        
        # Prepare context for LLM
        context = "\n\n".join(docs)
        system_prompt = """You are a helpful medical assistant. Use the provided medical information to answer the user's question accurately and concisely. 
        If the information is not sufficient, say so clearly. Always recommend consulting healthcare professionals for serious medical concerns."""
        
        user_prompt = f"Question: {query.query}\n\nMedical Information:\n{context}"
        
        # Generate response
        response_text = ""
        async for token in llm.stream_chat(system_prompt, user_prompt):
            response_text += token
        
        processing_time = time.time() - start_time
        
        return {
            "answer": response_text.strip(),
            "sources": "Medical Knowledge Base",
            "processing_time": round(processing_time, 3),
            "timestamp": datetime.now().isoformat(),
            "documents_retrieved": len(docs)
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "processing_time": round(processing_time, 3),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/rag/stream")
async def rag_stream_endpoint(query: Query):
    """Streaming RAG endpoint for real-time responses"""
    
    async def generate_stream():
        try:
            start_time = time.time()
            
            # Validate query
            if not query.query.strip():
                yield f"data: {{'error': 'Query cannot be empty'}}\n\n"
                return
            
            # Check if it's a medical query
            if not is_medical_query(query.query):
                yield f"data: {{'token': 'I\\'m a medical chatbot focused on health topics. Please ask a medical question.'}}\n\n"
                yield f"data: [DONE]\n\n"
                return
            
            # Retrieve documents
            docs = await store.async_retrieve(query.query)
            
            if not docs:
                yield f"data: {{'token': 'I couldn\\'t find relevant information in my medical knowledge base.'}}\n\n"
                yield f"data: [DONE]\n\n"
                return
            
            # Prepare context
            context = "\n\n".join(docs)
            system_prompt = """You are a helpful medical assistant. Use the provided medical information to answer the user's question accurately and concisely."""
            user_prompt = f"Question: {query.query}\n\nMedical Information:\n{context}"
            
            # Stream response
            async for token in llm.stream_chat(system_prompt, user_prompt):
                if token.strip():
                    yield f"data: {{'token': '{token.replace(chr(10), ' ').replace(chr(13), ' ')}'}}\n\n"
            
            # Send completion signal
            processing_time = time.time() - start_time
            yield f"data: {{'processing_time': {processing_time:.3f}, 'documents_retrieved': {len(docs)}}}\n\n"
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\n\n"
            yield f"data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        faiss_loaded = store.vectorstore is not None
        total_docs = 0
        index_size = 0
        
        if hasattr(store, 'vectorstore') and store.vectorstore is not None:
            try:
                total_docs = len(store.vectorstore.docstore._dict) if hasattr(store.vectorstore, 'docstore') else 0
                index_size = store.vectorstore.index.ntotal if hasattr(store.vectorstore, 'index') else 0
            except:
                pass
        
        return {
            "status": "operational",
            "faiss_loaded": faiss_loaded,
            "total_documents": total_docs,
            "index_size": index_size,
            "embedding_model": "models/embedding-001",
            "llm_model": "gemini-2.5-pro-latest",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Medical RAG API Server...")
    print("📋 Endpoints:")
    print("  • Health: GET /health")
    print("  • RAG: POST /rag")
    print("  • Stream: POST /rag/stream")
    print("  • Stats: GET /stats")
    print("  • Docs: GET /docs")
    
    port = int(os.getenv("PORT", 8000))  # Use 8001 to avoid conflicts
    print(f"🌐 Server will start on http://localhost:{port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        reload=False
    )
