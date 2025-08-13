# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BATCH_SIZE=32
ENV LEARNING_RATE=0.001
ENV N_LAYERS=4

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

# Install neural network specific dependencies
RUN pip install \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torchvision \
    tensorboard

# Copy neural network code
COPY src/neural /app/neural
COPY src/quantum/core /app/quantum/core

# Copy service files
COPY services/neural /app/services

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Start neural service
CMD ["python", "services/neural/main.py"]
