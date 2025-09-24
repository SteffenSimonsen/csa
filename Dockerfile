# Multi-stage build for production
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies to a virtual environment
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.12-slim AS runtime

# Install runtime dependencies (needed for ML libraries)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Make sure we use venv
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for models, logs, and cache
RUN mkdir -p checkpoints ml/models logs /tmp/cache /app/.cache /home/appuser && \
    chown -R appuser:appuser checkpoints ml/models logs /tmp/cache /app/.cache /home/appuser

# Set environment variables for caching
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV HF_HOME=/tmp/cache  
ENV MPLCONFIGDIR=/tmp/cache
ENV HOME=/home/appuser

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
