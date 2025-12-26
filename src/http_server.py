"""
HTTP API wrapper for Tax Code search
Provides REST endpoint for easy browser-based testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embeddings.vectorStore import get_vectorstore
from retrieval.search import search, initialize_bm25

# Initialize FastAPI
app = FastAPI(
    title="Tax Code Search API",
    description="Search the US Tax Code (Title 26 - Internal Revenue Code)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector store at startup
print("🚀 Starting Tax Code Search API...")
print("📦 Loading vector store...")
db = get_vectorstore("chroma_db")
print("🔍 Initializing hybrid search...")
initialize_bm25(db)
print("✅ API Ready!")

# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", example="SALT deduction limit")
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=20)
    alpha: Optional[float] = Field(0.5, description="Balance between semantic (1.0) and keyword (0.0) search", ge=0.0, le=1.0)

class SearchResult(BaseModel):
    text: str = Field(..., description="Content of the result")
    section: str = Field(..., description="Tax code section reference")
    page: int = Field(..., description="Page number in Title 26")
    score: float = Field(..., description="Relevance score (0-1, higher is better)")
    
class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    alpha: float

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "Tax Code Search API",
        "version": "1.0.0",
        "status": "operational",
        "description": "Search the US Tax Code (Title 26) using hybrid semantic + keyword search",
        "endpoints": {
            "search": "POST /search - Main search endpoint",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API documentation",
            "examples": "GET /examples - Example queries"
        },
        "documentation": "/docs"
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store": "loaded",
        "hybrid_search": "initialized",
        "ready": True
    }

@app.get("/examples")
def examples():
    """Example queries to try"""
    return {
        "examples": [
            {
                "query": "SALT deduction limit",
                "description": "Find information about state and local tax deduction limits",
                "expected": "Should find § 164 on page ~1343"
            },
            {
                "query": "senior citizen additional deduction",
                "description": "Find tax benefits for seniors",
                "expected": "Should find additional standard deduction provisions"
            },
            {
                "query": "Section 164",
                "description": "Direct section lookup",
                "expected": "Should return exact § 164"
            },
            {
                "query": "child tax credit",
                "description": "Find child tax credit information",
                "expected": "Should find § 24"
            },
            {
                "query": "401k contribution limits",
                "description": "Find retirement plan contribution rules",
                "expected": "Should find § 401"
            }
        ]
    }

@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    """
    Search the tax code
    
    Returns top k most relevant sections with scores.
    
    **Score interpretation:**
    - 0.80-1.00: Excellent match
    - 0.65-0.79: Good match
    - 0.50-0.64: Fair match
    - 0.00-0.49: Poor match
    
    **Alpha parameter:**
    - 0.0: Pure keyword search (good for exact terms)
    - 0.5: Balanced (default)
    - 1.0: Pure semantic search (good for concepts)
    """
    try:
        results = search(db, request.query, k=request.top_k, alpha=request.alpha)
        
        formatted_results = [
            SearchResult(
                text=r['text'],
                section=r['section'],
                page=r['page'],
                score=round(r['score'], 4)
            )
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total=len(formatted_results),
            alpha=request.alpha
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/search/{query}")
def search_get(query: str, top_k: int = 5, alpha: float = 0.5):
    """
    Search via GET request (convenience method)
    
    Example: /search/SALT%20deduction?top_k=3
    """
    try:
        results = search(db, query, k=top_k, alpha=alpha)
        
        formatted_results = [
            SearchResult(
                text=r['text'],
                section=r['section'],
                page=r['page'],
                score=round(r['score'], 4)
            )
            for r in results
        ]
        
        return SearchResponse(
            query=query,
            results=formatted_results,
            total=len(formatted_results),
            alpha=alpha
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))