from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

# Initialize splitter once
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try paragraph, then sentence, then word
)

def chunk_sections(sections: List[Dict]) -> List[Dict]:
    """
    Chunk sections into smaller pieces while preserving metadata.
    Each chunk retains section header, page number, and source.
    """
    chunks = []
    
    for sec in sections:
        text = sec.get("text", "")
        header = sec.get("header", "Unknown Section")
        page = sec.get("page", 0)
        
        # Split text into chunks
        texts = splitter.split_text(text)
        
        for i, chunk_text in enumerate(texts):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "section": header,
                    "page": page,
                    "source": "Title 26 - Internal Revenue Code",
                    "chunk_index": i,
                    "total_chunks": len(texts)  # Helpful for context
                }
            })
    
    return chunks