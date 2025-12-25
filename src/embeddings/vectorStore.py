# src/embeddings/vectorStore.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict
import os

def get_vectorstore(persist_dir: str = "chroma_db") -> Chroma:
    """
    Load existing vector store with HuggingFace embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )


# def create_vectorstore(chunks: List[Dict], persist_dir: str = "chroma_db") -> Chroma:
#     """
#     Create a new vector store from chunks.
    
#     Args:
#         chunks: List of dicts with 'content' and 'metadata' keys
#         persist_dir: Directory to save the vector store
        
#     Returns:
#         Chroma vector store instance
#     """
#     print(f"  Creating embeddings for {len(chunks)} chunks...")
    
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )
    
#     # Extract texts and metadatas
#     texts = [chunk["content"] for chunk in chunks]
#     metadatas = [chunk["metadata"] for chunk in chunks]
    
#     # Create vector store
#     vectorstore = Chroma.from_texts(
#         texts=texts,
#         embedding=embeddings,
#         metadatas=metadatas,
#         persist_directory=persist_dir
#     )
    
#     print(f"  Persisting to {persist_dir}...")
    
#     return vectorstore

# for docker setup
def create_vectorstore(chunks: List[Dict], persist_dir: str = "/app/chroma_db") -> Chroma:
    print(f"  Creating embeddings for {len(chunks)} chunks...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir
    )

    print(f"  Persisting vector store to {persist_dir}...")
    vectorstore.persist()

    return vectorstore


def vectorstore_exists(persist_dir: str = "/app/chroma_db") -> bool:
    return (
        os.path.exists(persist_dir)
        and os.path.isdir(persist_dir)
        and len(os.listdir(persist_dir)) > 0
    )
