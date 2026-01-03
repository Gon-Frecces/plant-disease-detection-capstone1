# Dockerfile - Pre-download PyTorch weights
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Pre-download PyTorch EfficientNet weights to cache
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -q https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth \
    -O /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth

# Install PyTorch CPU version
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu \
    torchvision==0.15.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL application files INCLUDING model
COPY . .

# Create model directory if needed
RUN mkdir -p models static/uploads

# Verify files exist
RUN echo "=== File Verification ===" && \
    echo "Model file:" && ls -lh models/plant_disease_classifier.pth && \
    echo "PyTorch weights cached:" && ls -lh /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV HOST=0.0.0.0
ENV TORCH_HOME=/tmp/torch  

# Create non-root user (Railway recommends)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /root/.cache 2>/dev/null || true

USER appuser

EXPOSE 5000

# Simple CMD
CMD ["python", "predict.py"]