# Use Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPTIMIZATION_STRATEGY=gradient
ENV MAX_ITERATIONS=1000
ENV CONVERGENCE_THRESHOLD=1e-6

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

# Install optimization specific dependencies
RUN pip install \
    scipy \
    scikit-learn \
    optuna \
    hyperopt \
    bayesian-optimization

# Copy optimization code
COPY src/optimization /app/optimization
COPY src/quantum/core /app/quantum/core
COPY src/quantum/utils /app/quantum/utils

# Copy service files
COPY services/optimization /app/services

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 5004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5004/health || exit 1

# Start optimization service
CMD ["python", "services/optimization/main.py"]
