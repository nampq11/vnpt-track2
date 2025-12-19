# VNPT Track 2 - Submission Dockerfile
# Optimized multi-stage build for minimal image size
# Based on: https://martynassubonis.substack.com/p/optimizing-docker-images-for-python

# ============================================================
# BUILDER STAGE
# ============================================================
FROM python:3.11-slim AS builder

# Upgrade pip with pinned version for reproducibility
RUN pip install --no-cache-dir --upgrade pip==24.1.1

WORKDIR /app

# Copy dependency files first (optimizes caching - these change less frequently)
COPY requirements.txt ./

# Install dependencies into virtual environment
# Using virtual env allows clean transfer to runtime stage
RUN python -m venv /app/.venv && \
    /app/.venv/bin/pip install --no-cache-dir --upgrade pip==24.1.1 && \
    /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# ============================================================
# RUNTIME STAGE
# ============================================================
FROM python:3.11-slim AS runtime

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Set PATH to use virtual environment (no need to activate)
ENV PATH="/code/.venv/bin:$PATH"

# Copy virtual environment from builder stage
# Positioned late to maximize cache efficiency
COPY --from=builder /app/.venv .venv

# Copy source code (changes frequently, so copy last)
COPY src/ src/
COPY predict.py .
COPY bin/ bin/

# Copy knowledge base (needed at runtime)
# Knowledge base is large (~435MB) but needed for RAG
COPY data/embeddings/ data/embeddings/

# Create config directory (can be mounted at runtime with -v)
RUN mkdir -p config

# Make submission script executable
RUN chmod +x bin/submission_inference.sh

# ============================================================
# EXECUTION
# Pipeline reads /code/private_test.json and outputs submission.csv & submission_time.csv
# ============================================================
CMD ["bash", "bin/submission_inference.sh"]

