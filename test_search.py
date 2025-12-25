# test_search.py
"""
Test the search functionality
"""

import sys
sys.path.append('src')

from embeddings.vectorStore import get_vectorstore
from retrieval.search import search

def test_search():
    """Test search with example queries"""
    
    print("=" * 80)
    print("TESTING TAX CODE SEARCH")
    print("=" * 80)
    
    # Load vector store
    print("\nLoading vector store...")
    db = get_vectorstore("chroma_db")
    print("✓ Vector store loaded")
    
    # Test queries
    test_queries = [
        ("SALT deduction limit", 3),
        ("state and local tax deduction", 3),
        ("senior citizen additional deduction", 3),
        ("section 164", 2)
    ]
    
    for query, k in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: '{query}' (top {k} results)")
        print("=" * 80)
        
        results = search(db, query, k=k)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Section: {result['section']}")
            print(f"Page: {result['page']}")
            print(f"Relevance: {result['score']:.3f}")
            print(f"\nContent preview:")
            print(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])
        
        print("\n")

if __name__ == "__main__":
    test_search()