# services/encoder/Dockerfile
#
# WHY python:3.11-slim AND NOT python:3.11:
#   The full python:3.11 image is ~900MB — it includes compilers, dev headers,
#   documentation, and dozens of tools you never need at runtime.
#   python:3.11-slim is ~130MB — just the runtime.
#
#   We do need build tools temporarily to compile some Python packages
#   (onnxruntime has C extensions). We install them, build, then remove them.
#   This is called a multi-stage awareness pattern — using build deps only
#   when needed.
#
# FINAL IMAGE SIZE TARGET: ~800MB
#   - python:3.11-slim base: 130MB
#   - onnxruntime: ~400MB (it's big — ONNX runtime is a full inference engine)
#   - clip + torchvision: ~150MB (we need torchvision for preprocessing)
#   - Our code: <1MB
#   - ONNX model files: ~90MB (mounted as volume, not baked in)
#
# WHY NOT ALPINE (even smaller base):
#   Alpine uses musl libc instead of glibc.
#   onnxruntime ships pre-compiled wheels built against glibc.
#   Using Alpine would require compiling onnxruntime from source: 2+ hours.
#   Not worth it for a <100MB size difference.

FROM python:3.11-slim

WORKDIR /app

# Install system deps needed to build Python packages with C extensions
# --no-install-recommends: only install exactly what's listed, not suggested packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/ removes the package cache after install
# Every RUN command creates a Docker layer. The cleanup MUST be in the same
# RUN to actually reduce layer size. If you put it in a separate RUN,
# the cache files are already baked into the previous layer.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir: don't store pip's download cache in the image
# pip normally caches to speed up repeated installs, but in Docker
# we build once and run many times — cache just wastes space.

# Copy only the application code (not scripts, not embeddings)
COPY main.py .

# Create directory for model files
# Actual models are mounted via Docker volume — NOT baked into the image.
# WHY NOT BAKE IN:
#   If models are in the image, every model update requires rebuilding the image.
#   With volumes, you can update models without touching Docker.
#   Also: 90MB model in image = 90MB transferred on every docker pull.
RUN mkdir -p models

# Port 8001: encoder service (internal, not exposed to host directly)
EXPOSE 8001

# HEALTHCHECK: Docker polls this to know if the service is ready.
# --interval: check every 30s
# --timeout: fail if no response in 10s
# --start-period: wait 60s before starting checks (model loading takes time)
# --retries: mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

# Run with uvicorn (ASGI server for FastAPI)
# --host 0.0.0.0: listen on all interfaces (needed inside Docker)
# --port 8001: encoder port
# --workers 1: ONE worker. Why?
#   ONNX sessions are NOT safely forkable.
#   Multiple workers would each load the 90MB model into RAM.
#   For CPU-bound inference, multiple workers don't help — use async instead.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]