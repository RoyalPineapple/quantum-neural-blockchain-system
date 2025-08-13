# Use Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRADING_FREQUENCY=1.0
ENV RISK_TOLERANCE=0.1
ENV INITIAL_CAPITAL=1000000

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

# Install financial specific dependencies
RUN pip install \
    pandas \
    numpy \
    scipy \
    statsmodels \
    yfinance \
    ta-lib \
    empyrical \
    zipline-reloaded

# Copy financial code
COPY src/applications/financial /app/applications/financial
COPY src/quantum/core /app/quantum/core
COPY src/neural/core /app/neural/core
COPY src/blockchain/core /app/blockchain/core

# Copy service files
COPY services/financial /app/services

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5003/health || exit 1

# Start financial service
CMD ["python", "services/financial/main.py"]
