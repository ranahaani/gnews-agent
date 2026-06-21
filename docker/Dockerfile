# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

# System deps: build tools for sentence-transformers/chromadb wheels that
# may need compilation on slim, plus libgomp1 for the embedding model.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package + the two extras most users want at runtime. Skipping
# `lance` and `qdrant` keeps the image lean — they slot in via a derived
# image when needed.
COPY pyproject.toml LICENSE ./
COPY gnews_agent ./gnews_agent
COPY mcp_server  ./mcp_server
RUN pip install --upgrade pip \
 && pip install ".[openai,fulltext]"

# Pre-warm the sentence-transformers model so first ingest doesn't pay the
# ~80MB download cold-start. Done in the build, not at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Persistent state lives here; docker-compose mounts a host volume on top.
ENV GNEWS_AGENT_HOME=/data
VOLUME ["/data"]

EXPOSE 8000
ENTRYPOINT ["gnews-agent"]
CMD ["serve", "--transport", "http", "--port", "8000", "--db-path", "/data/news.db", "--vector-path", "/data/chroma"]
