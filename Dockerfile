FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set torch cache to writable location
ENV TORCH_HOME=/app/torch_cache

# Create torch cache dir
RUN mkdir -p $TORCH_HOME/hub/checkpoints

# Pre-download EfficientNet weights
RUN wget -q https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth \
    -O $TORCH_HOME/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth

# Install PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu \
    torchvision==0.15.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Ensure required dirs exist
RUN mkdir -p models static/uploads

# Verify files
RUN echo "=== Verification ===" && \
    ls -lh models && \
    ls -lh $TORCH_HOME/hub/checkpoints

# Railway-friendly env
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

EXPOSE 5000

# DO NOT SWITCH USER (Railway does sandboxing already)
CMD ["python", "predict.py"]
