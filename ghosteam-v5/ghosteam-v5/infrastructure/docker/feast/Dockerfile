FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Feast and dependencies
RUN pip install --no-cache-dir \
    feast[redis,postgres]>=0.34.0 \
    pandas \
    numpy

# Create feast user
RUN useradd -m -u 1000 feast

# Create directories
RUN mkdir -p /feast_repo && chown feast:feast /feast_repo

USER feast

WORKDIR /feast_repo

# Expose port
EXPOSE 6566

# Default command
CMD ["feast", "serve", "--host", "0.0.0.0", "--port", "6566"]
