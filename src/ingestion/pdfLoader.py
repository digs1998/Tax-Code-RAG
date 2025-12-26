from langchain_community.document_loaders import PyPDFLoader
from typing import List

def load_pdf(path: str):
    """
    Load PDF and return documents with page content and metadata.
    Returns list of Document objects from langchain_community.
    """
    loader = PyPDFLoader(path)
    docs = loader.load()
    
    # Clean up text - PDFs often have formatting issues
    for doc in docs:
        # Remove excessive whitespace
        doc.page_content = " ".join(doc.page_content.split())
        # Ensure page number is 1-indexed (more intuitive)
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
    
    return docs