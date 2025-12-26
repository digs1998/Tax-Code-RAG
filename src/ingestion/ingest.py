"""
Complete ingestion pipeline for tax code PDF
Run this once to build the vector store
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdfLoader import load_pdf
from ingestion.sectionParser import parse_documents
from ingestion.chunker import chunk_sections
from embeddings.vectorStore import create_vectorstore, vectorstore_exists

def ingest_tax_code(
    pdf_path: str = "data/usc26@119-59.pdf",
    persist_dir: str = "/app/chroma_db",
    force_rebuild: bool = False
):
    """
    Complete ingestion pipeline:
    1. Load PDF
    2. Parse into sections
    3. Chunk sections
    4. Create vector store
    """
    
    # Check if already exists
    if vectorstore_exists(persist_dir) and not force_rebuild:
        print(f"✓ Vector store already exists at {persist_dir}")
        print("  Use --rebuild flag to force rebuild")
        return
    
    print("=" * 80)
    print("STARTING TAX CODE INGESTION PIPELINE")
    print("=" * 80)
    
    # Step 1: Load PDF
    print("\n[1/4] Loading PDF...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    
    docs = load_pdf(pdf_path)
    print(f"  ✓ Loaded {len(docs)} pages")
    
    # Step 2: Parse into sections
    print("\n[2/4] Parsing sections...")
    sections = parse_documents(docs)
    print(f"  ✓ Parsed {len(sections)} sections")
    
    # Print sample sections
    print("\n  Sample sections:")
    for sec in sections[:5]:
        header = sec['header'][:80] + "..." if len(sec['header']) > 80 else sec['header']
        print(f"    - {header} (page {sec['page']})")
    
    # Step 3: Chunk sections
    print("\n[3/4] Chunking sections...")
    chunks = chunk_sections(sections)
    print(f"  ✓ Created {len(chunks)} chunks")
    
    # Print chunk statistics
    avg_length = sum(len(c["content"]) for c in chunks) / len(chunks) if chunks else 0
    print(f"  ✓ Average chunk length: {avg_length:.0f} characters")
    
    # Step 4: Create vector store
    print("\n[4/4] Creating vector store...")
    print("  (This may take several minutes - downloading embeddings model...)")
    
    try:
        vectorstore = create_vectorstore(chunks, persist_dir)
        print(f"  ✓ Vector store created at {persist_dir}")
    except Exception as e:
        print(f"  ✗ Error creating vector store: {e}")
        raise
    
    # Verify
    print("\n" + "=" * 80)
    print("INGESTION COMPLETE!")
    print("=" * 80)
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Total sections: {len(sections)}")
    print(f"Vector store location: {persist_dir}")
    print("\nYou can now run queries using the search module.")
    print("Test with: python test_search.py")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest tax code PDF")
    parser.add_argument(
        "--pdf",
        default="data/usc26@119-59.pdf",
        help="Path to tax code PDF"
    )
    parser.add_argument(
        "--output",
        default="chroma_db",
        help="Output directory for vector store"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if vector store exists"
    )
    
    args = parser.parse_args()
    
    try:
        ingest_tax_code(
            pdf_path=args.pdf,
            persist_dir=args.output,
            force_rebuild=args.rebuild
        )
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        sys.exit(1)