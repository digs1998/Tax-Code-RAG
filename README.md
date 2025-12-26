# Tax Code RAG – MCP & HTTP Server

This repository contains a Retrieval-Augmented Generation (RAG) system built over U.S. tax code documents. The core logic is exposed via an **MCP server (STDIO-based)**, with supporting scripts for ingestion and local evaluation. The system uses **ChromaDB** for vector storage and supports configurable retrieval parameters (e.g., `k`).


## High-Level Architecture

* **Ingestion Pipeline**: Parses tax code documents, generates embeddings, and stores them in ChromaDB
* **Vector Store**: ChromaDB with local persistence (`chroma_db/`)
* **MCP Server**: Primary interface for querying the RAG pipeline using MCP (STDIO)
* **Inspector UI**: Used to interactively test queries via browser

The design intentionally separates **ingestion**, **retrieval**, and **serving**, making it easier to adapt or extend.


## Prerequisites

* Python 3.12
* GitHub
* Docker
* [MCP documentation](https://modelcontextprotocol.io/docs/tools/inspector)

## Local Setup (Python venv)

### 1. Create and activate a virtual environment

```
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

## Ingestion: Build the Vector Store

Before running the server, embeddings must be generated and stored locally.

```
python src/ingestion/ingest.py
```

This will:

* Parse the tax code source documents
* Generate embeddings
* Persist the vector store to `./chroma_db/`

> **Note**: `chroma_db/` is generated at runtime and is not committed to Git.

## Running the MCP Server

Start the MCP server using:

```
mcp-inspector python src/main.py
```

On startup, the logs will print a **local inspection URL and access key**.

### Using the MCP Inspector

1. Copy the URL and key from the logs
2. Open the URL in your browser
3. Paste the key when prompted
4. Submit queries against the tax code

You can experiment with:

* Different natural language questions
* Different `k` values (number of retrieved chunks)

This allows you to directly observe retrieval quality and grounding behavior.


## Configuration Notes

Common parameters can be adjusted via environment variables or config files:

* `CHROMA_PATH` – location of the vector store (default: `./chroma_db`)
* `TOP_K` – number of retrieved documents per query
* Embedding / LLM provider settings

These defaults are chosen to make local evaluation simple.


## Docker Setup (Optional)

The project can also be run using Docker for a reproducible environment.

### Build the image

```bash
docker compose up --build -d .
```

### Run the MCP server

```bash
docker run -p 8010:8010 -v $(pwd)/chroma_db:/app/chroma_db tax-code-mcp \
  python src/main.py
```

> The `chroma_db` directory is mounted as a volume so embeddings persist across runs.

---

## Notes on Design Choices

* The MCP server is the **primary interface**; HTTP adapters can be layered on top if needed
* Vector DB artifacts are generated locally to avoid committing large derived files
* The system favors clarity and reproducibility over production hardening


## Extending the Project

Common extensions include:

* Adding an HTTP/FastAPI adapter on top of the MCP server
* Supporting additional document sources
* Swapping vector stores or embedding models
* Deploying the HTTP adapter to a managed service


## Summary

This repository demonstrates:

* End-to-end RAG over tax code data
* Clean separation between ingestion, retrieval, and serving
* MCP-based interaction with configurable retrieval parameters

It is intended to be easy to run locally, inspect interactively, and adapt for further experimentation.
