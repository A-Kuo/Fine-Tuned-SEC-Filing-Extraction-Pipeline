# FinDocAnalyzer Production Dockerfile
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Production
# ============================================================================
FROM python:3.11-slim as production

WORKDIR /app

# Copy installed packages from builder. System-wide (not --user/~/.local) so
# they're importable regardless of which user runs the container -- a
# --user install under root's home is invisible to the non-root appuser
# this image switches to below, which made every third-party import
# (httpx first, alphabetically/positionally, but all of them) fail with
# ModuleNotFoundError the first time this image was actually run end-to-end.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy application code
COPY src/ ./src/
COPY training/ ./training/
COPY serving/ ./serving/
COPY evaluation/ ./evaluation/
COPY monitoring/ ./monitoring/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY config.yaml .
COPY pyproject.toml .
COPY Makefile .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/results /app/logs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Health check. Uses urllib (stdlib) rather than `requests` -- the latter is
# not a declared dependency (only httpx is), so this always failed with
# ModuleNotFoundError, permanently keeping the container "unhealthy" no
# matter how well the app itself was running.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Expose ports
EXPOSE 8000 8001

# Default command
CMD ["python", "-m", "serving.api"]
