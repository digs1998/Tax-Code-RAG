# src/retrieval/search.py - Hybrid Search
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import numpy as np

# Global BM25 index (initialized once)
_bm25_index = None
_bm25_docs = None
_bm25_metadata = None

def initialize_bm25(db: Chroma):
    """
    Initialize BM25 index from ChromaDB.
    Call this once when loading the vector store.
    """
    global _bm25_index, _bm25_docs, _bm25_metadata
    
    if _bm25_index is not None:
        return  # Already initialized
    
    print("Initializing BM25 index for hybrid search...", flush=True)
    
    # Get all documents from ChromaDB
    # Note: This loads everything into memory - fine for 32k chunks
    try:
        all_results = db.similarity_search("", k=100000)  # Get all docs
    except:
        # Fallback: get a large sample
        all_results = db.similarity_search("tax code", k=10000)
    
    _bm25_docs = []
    _bm25_metadata = []
    
    for doc in all_results:
        _bm25_docs.append(doc.page_content)
        _bm25_metadata.append({
            'doc': doc,
            'section': doc.metadata.get('section', 'Unknown'),
            'page': doc.metadata.get('page', 0)
        })
    
    # Tokenize for BM25
    tokenized_docs = [doc.lower().split() for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenized_docs)
    
    print(f"✓ BM25 index ready with {len(_bm25_docs)} documents", flush=True)


def search_semantic(db: Chroma, query: str, k: int = 10) -> List[Dict]:
    """Pure semantic search using embeddings"""
    results = db.similarity_search_with_score(query, k=k)
    
    formatted_results = []
    for doc, distance in results:
        similarity_score = 1.0 / (1.0 + distance)
        
        formatted_results.append({
            "text": doc.page_content,
            "section": doc.metadata.get("section", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "source": doc.metadata.get("source", ""),
            "score": float(similarity_score),
            "distance": float(distance),
            "metadata": doc.metadata
        })
    
    return formatted_results


def search_bm25(query: str, k: int = 10) -> List[Dict]:
    """Pure keyword search using BM25"""
    global _bm25_index, _bm25_docs, _bm25_metadata
    
    if _bm25_index is None:
        return []
    
    # Tokenize query
    tokenized_query = query.lower().split()
    
    # Get BM25 scores
    scores = _bm25_index.get_scores(tokenized_query)
    
    # Get top k indices
    top_indices = np.argsort(scores)[::-1][:k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include matches with score > 0
            meta = _bm25_metadata[idx]
            results.append({
                "text": _bm25_docs[idx],
                "section": meta['section'],
                "page": meta['page'],
                "source": "Title 26 - Internal Revenue Code",
                "score": float(scores[idx]),
                "metadata": meta['doc'].metadata
            })
    
    return results


def search(db: Chroma, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict]:
    """
    Hybrid search combining semantic and keyword search with proper score normalization.
    
    Args:
        db: ChromaDB vector store instance
        query: Search query string
        k: Number of results to return
        alpha: Weight for semantic vs keyword (0=pure BM25, 1=pure semantic, 0.5=balanced)
        
    Returns:
        List of dicts with text, section, page, score (0-1 range), and metadata
    """
    # Initialize BM25 if needed
    if _bm25_index is None:
        initialize_bm25(db)
    
    # Get results from both methods (get more, then combine)
    semantic_results = search_semantic(db, query, k=k*3)
    bm25_results = search_bm25(query, k=k*3)
    
    # Normalize scores to 0-1 range for each method
    def normalize_scores(results):
        if not results or len(results) < 2:
            return results
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            for r in results:
                r['normalized_score'] = 1.0
            return results
        
        for r in results:
            r['normalized_score'] = (r['score'] - min_score) / (max_score - min_score)
        return results
    
    semantic_results = normalize_scores(semantic_results)
    bm25_results = normalize_scores(bm25_results)
    
    # Combine results using weighted normalized scores
    combined_scores = {}
    
    # Add semantic results
    for result in semantic_results:
        key = result['text'][:100]  # Use text prefix as key
        if key not in combined_scores:
            combined_scores[key] = {
                'result': result,
                'semantic_score': 0,
                'bm25_score': 0,
                'count': 0
            }
        combined_scores[key]['semantic_score'] = result.get('normalized_score', result['score'])
        combined_scores[key]['count'] += 1
    
    # Add BM25 results
    for result in bm25_results:
        key = result['text'][:100]
        if key not in combined_scores:
            combined_scores[key] = {
                'result': result,
                'semantic_score': 0,
                'bm25_score': 0,
                'count': 0
            }
        combined_scores[key]['bm25_score'] = result.get('normalized_score', result['score'])
        combined_scores[key]['count'] += 1
    
    # Calculate final weighted score (guaranteed 0-1 range)
    for key, data in combined_scores.items():
        # Weighted combination of normalized scores
        final_score = (alpha * data['semantic_score']) + ((1 - alpha) * data['bm25_score'])
        
        # Boost if document appears in both result sets
        if data['count'] > 1:
            final_score *= 1.2  # 20% boost for consensus
        
        # Clamp to 0-1 range (just to be safe)
        final_score = min(1.0, max(0.0, final_score))
        
        data['final_score'] = final_score
    
    # Sort by final score
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x['final_score'],
        reverse=True
    )
    
    # Format final results with guaranteed 0-1 score range
    final_results = []
    for item in sorted_results[:k]:
        result = item['result'].copy()
        result['score'] = float(item['final_score'])  # Normalized 0-1 score
        result['semantic_component'] = float(item['semantic_score'])
        result['bm25_component'] = float(item['bm25_score'])
        final_results.append(result)
    
    return final_results


def search_with_boost(db: Chroma, query: str, k: int = 5) -> List[Dict]:
    """
    Hybrid search with automatic boost for section number queries.
    Detects if query contains section numbers and adjusts weights.
    """
    import re
    
    # Detect if query contains section numbers
    has_section = bool(re.search(r'(§|section|sec\.?)\s*\d+', query.lower()))
    
    # Boost keyword search for section number queries
    alpha = 0.3 if has_section else 0.5  # More weight on BM25 for section queries
    
    return search(db, query, k=k, alpha=alpha)