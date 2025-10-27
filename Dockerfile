# Multi-stage build for SPIDERWEB ML-as-a-Service Platform
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (optional, for building native extensions)
# Uncomment if you need to build Rust components in container
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"

# Copy Python dependencies
COPY pyproject.toml uv.lock* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    anthropic>=0.54.0 \
    flask>=3.1.1 \
    openai>=1.88.0 \
    requests>=2.32.4 \
    gunicorn>=21.2.0

# Copy application code
COPY demo_server.py ./
COPY examples/ ./examples/
COPY docs/ ./docs/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "demo_server:app"]
