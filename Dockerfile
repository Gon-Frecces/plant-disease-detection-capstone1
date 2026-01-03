# Stage 1: Builder (will be discarded)
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies only
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create wheel packages (separates build from runtime)
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt


# Stage 2: Runtime (final image - much smaller)
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies ONLY if needed
# Remove if not using OpenCV, etc.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels
COPY requirements.txt .

# Install from wheels (no build tools needed)
RUN pip install --no-cache --no-index /wheels/* && \
    rm -rf /wheels

# Create non-root user for security
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

# Switch to non-root user
USER user

# Copy only necessary application files
COPY --chown=user:user predict.py .
# DO NOT copy model files if they're large!

EXPOSE 5000

CMD ["python", "predict.py"]