# Use CUDA base image for GPU support
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV N_QUBITS=8
ENV ERROR_THRESHOLD=0.001
ENV CORRECTION_ROUNDS=3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install quantum-specific dependencies
RUN pip3 install \
    qiskit \
    cirq \
    pyquil \
    pennylane

# Copy quantum service code
COPY src/quantum /app/quantum
COPY src/optimization/circuits /app/optimization/circuits

# Copy service files
COPY services/quantum /app/services

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start quantum service
CMD ["python3", "services/quantum/main.py"]
