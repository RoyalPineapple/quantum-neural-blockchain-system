# Use Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DIFFICULTY=4
ENV BLOCK_SIZE=100
ENV MINING_REWARD=10.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install blockchain specific dependencies
RUN pip install \
    web3 \
    eth-utils \
    eth-keys \
    eth-hash \
    eth-typing

# Copy blockchain code
COPY src/blockchain /app/blockchain
COPY src/quantum/core /app/quantum/core
COPY src/quantum/utils /app/quantum/utils

# Copy service files
COPY services/blockchain /app/services

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# Start blockchain service
CMD ["python", "services/blockchain/main.py"]
