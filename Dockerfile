# Dockerfile - with torch CPU version
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CPU version first
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu \
    torchvision==0.15.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models /app/static/uploads && \
    chown -R appuser:appuser /app

USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

ENV PYTHONUNBUFFERED=1
ENV MODEL_URL="https://drive.google.com/uc?id=1NeGIjmkg_OroQerpVnlho5nKWqf51oI2"
ENV PORT=5000

EXPOSE 5000

CMD ["python", "predict.py"]