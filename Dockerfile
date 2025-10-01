# Multi-stage build for production
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install ONLY api dependencies
RUN uv sync --frozen --no-dev --group api

# Production stage
FROM python:3.12-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy venv from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy only necessary application code
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser db/ ./db/
COPY --chown=appuser:appuser ml/ ./ml/

# Create directories
RUN mkdir -p logs /tmp/cache /tmp/models /home/appuser && \
    chown -R appuser:appuser logs /tmp/cache /tmp/models /home/appuser

ENV HF_HOME=/tmp/cache
ENV HOME=/home/appuser

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
