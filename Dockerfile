# Dockerfile - WITH model included
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

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

# Verify model is included
RUN echo "Model file check:" && \
    if [ -f "models/plant_disease_classifier.pth" ]; then \
        ls -lh models/plant_disease_classifier.pth && \
        echo "✓ Model file found"; \
    else \
        echo "❌ Model file NOT found!" && exit 1; \
    fi

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV HOST=0.0.0.0

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:5000/health', timeout=5); exit(0 if r.status_code == 200 else 1)"

CMD ["python", "predict.py"]