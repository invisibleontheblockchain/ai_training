# Railway deployment Dockerfile - Simplified for free tier
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data/models data/features data/processed mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=${DATABASE_URL}
ENV REDIS_URL=${REDIS_URL}
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV SECRET_KEY=${SECRET_KEY:-default-secret-key}
ENV DEBUG=false
ENV ENVIRONMENT=production

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
CMD ["python", "-m", "uvicorn", "src.serving.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
