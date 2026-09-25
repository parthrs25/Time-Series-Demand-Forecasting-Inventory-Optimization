# Production Dockerfile for Retail Multi-Store Hierarchical Demand Forecaster
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, and scripts
COPY config/ config/
COPY src/ src/
COPY run_pipeline.py .

# Expose MLflow default port
EXPOSE 5000

# Default entrypoint runs the full end-to-end MLOps pipeline
CMD ["python", "run_pipeline.py", "--stage", "all"]
