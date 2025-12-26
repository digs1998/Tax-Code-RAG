FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Build vector store (done once, shared by both images)
RUN echo "Building vector store..." && \
    mkdir -p /app/chroma_db && \
    python src/ingestion/ingest.py --pdf data/usc26@119-59.pdf --output /app/chroma_db && \
    echo "✓ Vector store built successfully"

# Verify vector store was created
RUN test -d /app/chroma_db && [ "$(ls -A /app/chroma_db)" ] && \
    echo "✓ Vector store verified" || \
    (echo "✗ Vector store empty!" && exit 1)

# Set common environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ============================================================================
# Stage 2: MCP Server (stdio mode)
# ============================================================================
FROM base AS mcp

# No ports needed (uses stdio)
CMD ["python", "src/main.py"]

# ============================================================================
# Stage 3: HTTP API Server
# ============================================================================
FROM base AS http

# HTTP-specific config
ENV PORT=8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "src/api/http_server.py"]