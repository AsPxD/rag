import os
import hashlib
import time
import json
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
from datetime import datetime

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('medical_chatbot.log')  # File output
    ]
)

load_dotenv()

app = FastAPI(title="Medical RAG Chatbot API", version="1.0.0")
executor = ThreadPoolExecutor(max_workers=4)

# Add CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache with TTL and LRU eviction
class InMemoryCache:
    def __init__(self, max_size=1000, ttl=3600):
        self.cache = OrderedDict()
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            del self.timestamps[oldest]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.cache.move_to_end(key)

# Initialize cache
cache = InMemoryCache()

# Load API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not set in .env file.")
genai.configure(api_key=api_key)

# Paths
PDF_PATH = "medical_data.pdf"
FAISS_PATH = "faiss_index"

# Perfect JSON formatter for terminal output
def print_beautiful_json(data, title=""):
    """Print beautifully formatted JSON to terminal"""
    print("\n" + "="*80)
    if title:
        print(f"🔍 {title}")
    print("="*80)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print("="*80 + "\n")

# Request/Response logger
def log_request_response(request_data, response_data, processing_time, cache_hit=False):
    """Log detailed request/response information"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data,
        "processing_time_seconds": round(processing_time, 3),
        "cache_hit": cache_hit,
        "cache_size": len(cache.cache)
    }
    
    print_beautiful_json(log_data, "API REQUEST/RESPONSE LOG")
    logging.info(f"Query processed: '{request_data.get('query', 'N/A')}' | Time: {processing_time:.3f}s | Cache: {'HIT' if cache_hit else 'MISS'}")

# Cache functions
def get_cache_key(query: str):
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

def get_cached_response(query: str):
    key = get_cache_key(query)
    return cache.get(key)

def cache_response(query: str, answer: str, sources: str):
    key = get_cache_key(query)
    cache.set(key, {"answer": answer, "sources": sources})

# Optimized data loading
def load_and_chunk_data():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}.")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    logging.info(f"Loaded and chunked {len(docs)} documents.")
    return docs

# Optimized vector store
def setup_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists(FAISS_PATH):
        try:
            vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            logging.info("Loaded optimized FAISS index.")
            return vectorstore
        except Exception as e:
            logging.error(f"Failed to load FAISS index: {e}. Recreating...")
    docs = load_and_chunk_data()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_PATH)
    return vectorstore

# Fast medical query check
def is_medical_query_fast(query: str):
    query_lower = query.lower()
    
    non_medical_patterns = {
        "weather", "time", "joke", "story", "math", "programming", 
        "sports", "entertainment", "politics", "recipe", "hello", "hi"
    }
    if any(pattern in query_lower for pattern in non_medical_patterns):
        return False
    
    medical_keywords = [
        "seizure", "epilepsy", "medicine", "doctor", "health", "symptom", 
        "treatment", "pain", "disease", "medical", "hospital", "drug", 
        "therapy", "diagnosis", "medication", "condition"
    ]
    if any(keyword in query_lower for keyword in medical_keywords):
        return True
    
    return is_medical_query(query)

def is_medical_query(query: str):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            f"Is this medical/health related? Answer only 'yes' or 'no': {query}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=10,
                temperature=0
            )
        )
        return response.text.strip().lower() == 'yes'
    except:
        return False

# Async functions
async def async_qa_chain_invoke(query: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: qa_chain.invoke({"query": query}))

async def async_fallback_to_gemini(query: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fallback_to_gemini, query)

def fallback_to_gemini(query: str):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            f"You are a helpful doctor specializing in seizures. Answer briefly in simple language. Start with: 'I'm not a real doctor, so please see a professional for advice.' Question: {query}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300, 
                temperature=0.1
            )
        )
        return response.text, "Direct from Gemini AI"
    except Exception as e:
        logging.error(f"Fallback error: {e}")
        return "Sorry, I couldn't find info. Please try again or ask a doctor.", "Error"

# Global variables
vectorstore = None
qa_chain = None

@app.on_event("startup")
async def startup_event():
    global vectorstore, qa_chain
    
    startup_log = {
        "event": "APPLICATION_STARTUP",
        "timestamp": datetime.now().isoformat(),
        "status": "INITIALIZING"
    }
    print_beautiful_json(startup_log, "MEDICAL CHATBOT STARTUP")
    
    vectorstore = setup_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    prompt_template = """You are a helpful doctor specializing in seizures. Answer briefly in simple language. Always start with: "I'm not a real doctor, so please see a professional for advice."

Context: {context}
Question: {question}
Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.1, 
            max_output_tokens=300
        ),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    startup_complete = {
        "event": "APPLICATION_STARTUP",
        "timestamp": datetime.now().isoformat(),
        "status": "COMPLETED",
        "components_loaded": [
            "Vector Store (FAISS)",
            "QA Chain (LangChain)",
            "LLM (Gemini 1.5 Flash)",
            "Cache System",
            "CORS Middleware"
        ]
    }
    print_beautiful_json(startup_complete, "STARTUP COMPLETED SUCCESSFULLY")

class Query(BaseModel):
    query: str

class HealthStatus(BaseModel):
    status: str = "healthy"
    cache_size: int = 0
    timestamp: str = ""

@app.get("/health")
async def health_check():
    """Health endpoint for React to check if backend is ready"""
    health_data = HealthStatus(
        status="healthy" if qa_chain is not None else "loading",
        cache_size=len(cache.cache),
        timestamp=datetime.now().isoformat()
    )
    
    print_beautiful_json(health_data.dict(), "HEALTH CHECK")
    return health_data

@app.post("/ask")
async def ask(query: Query):
    """Main endpoint for React to send medical queries"""
    start_time = time.time()
    cache_hit = False
    
    try:
        # Check cache first
        cached = get_cached_response(query.query)
        if cached:
            cache_hit = True
            processing_time = time.time() - start_time
            
            response_data = {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "cached": True,
                "processing_time": round(processing_time, 3),
                "timestamp": datetime.now().isoformat()
            }
            
            log_request_response(
                {"query": query.query}, 
                response_data, 
                processing_time, 
                cache_hit=True
            )
            
            return response_data
        
        # Fast medical check
        if not is_medical_query_fast(query.query):
            response_data = {
                "answer": "Sorry, I'm a medical chatbot focused on health topics, especially seizures. Please ask a medical question.", 
                "sources": "N/A",
                "cached": False,
                "processing_time": round(time.time() - start_time, 3),
                "timestamp": datetime.now().isoformat()
            }
            cache_response(query.query, response_data["answer"], response_data["sources"])
            
            log_request_response(
                {"query": query.query}, 
                response_data, 
                time.time() - start_time, 
                cache_hit=False
            )
            
            return response_data
        
        # Process medical query
        result = await async_qa_chain_invoke(query.query)
        if not result["source_documents"] or not result["result"].strip():
            answer, sources = await async_fallback_to_gemini(query.query)
        else:
            sources = ", ".join([doc.metadata.get("source", "Unknown") for doc in result["source_documents"]])
            answer = result["result"]
        
        processing_time = time.time() - start_time
        
        # Perfect formatted response
        response_data = {
            "answer": answer,
            "sources": sources,
            "cached": False,
            "processing_time": round(processing_time, 3),
            "timestamp": datetime.now().isoformat(),
            "method": "RAG" if result["source_documents"] else "Fallback",
            "documents_retrieved": len(result.get("source_documents", []))
        }
        
        # Cache response
        cache_response(query.query, answer, sources)
        
        # Log with beautiful formatting
        log_request_response(
            {"query": query.query}, 
            response_data, 
            processing_time, 
            cache_hit=False
        )
        
        return response_data
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_response = {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "processing_time": round(processing_time, 3)
        }
        
        print_beautiful_json(error_response, "ERROR OCCURRED")
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/clear")
async def clear_cache():
    """Endpoint to clear cache if needed"""
    cache.cache.clear()
    cache.timestamps.clear()
    
    clear_log = {
        "event": "CACHE_CLEARED",
        "timestamp": datetime.now().isoformat(),
        "message": "Cache cleared successfully"
    }
    print_beautiful_json(clear_log, "CACHE MANAGEMENT")
    
    return clear_log

if __name__ == "__main__":
    import uvicorn
    
    startup_info = {
        "server": "Medical RAG Chatbot API",
        "host": "0.0.0.0",
        "port": 8000,
        "docs_url": "http://localhost:8000/docs",
        "health_check": "http://localhost:8000/health"
    }
    print_beautiful_json(startup_info, "STARTING SERVER")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
