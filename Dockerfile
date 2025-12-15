# Titan Shield RAG System - Docker Runtime
# Constraint: NO external models (OpenAI, HuggingFace), ONLY VNPT APIs + standard Python libs

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src ./src
COPY data ./data
COPY src/artifacts ./src/artifacts

# Install uv and dependencies
RUN pip install --no-cache-dir uv && \
    uv sync --frozen

# Create results directory
RUN mkdir -p results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Health check script
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command: Run inference
CMD ["python", "-m", "asyncio", "-c", "import asyncio; from predict import main; asyncio.run(main())"]

# Runtime configuration
EXPOSE 8000

# Labels
LABEL maintainer="VNPT AI Team"
LABEL description="Titan Shield RAG System - Vietnamese QA Agent"
LABEL version="2.0.0"

