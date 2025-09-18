# AGI Agent Ecosystem - Railway Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY railway_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r railway_requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p agents logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV RAILWAY_ENVIRONMENT=production

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start script
CMD ["python", "start_agi_system.py"]
