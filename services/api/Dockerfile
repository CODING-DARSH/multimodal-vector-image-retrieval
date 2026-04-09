# services/api/Dockerfile
#
# WHY THIS IS SEPARATE FROM THE ENCODER:
#   If they were one container:
#   - Restart API → also restarts encoder → 3s model reload on every code change
#   - Scale horizontally → each replica carries the 90MB model in RAM
#   - One crash takes down both search logic AND inference
#
#   Separate containers = independent restart, scale, update, and failure domains.
#
# THIS CONTAINER IS LIGHTER than the encoder:
#   - No onnxruntime (that's the encoder's job)
#   - Needs faiss-cpu, whisper, httpx (for calling encoder)
#   - Target size: ~600MB

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
     git \
    # ffmpeg is needed by Whisper to decode audio files (mp3, wav, webm, etc.)
    # Without it, Whisper can only handle raw PCM.
    # Size cost: ~80MB — worth it for voice search capability.
    
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt
COPY main.py .

# Create directories for runtime data
# embeddings/ and data/ are mounted as volumes — not baked in
RUN mkdir -p embeddings data images

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 2 workers for the API (it's I/O bound — waiting on encoder HTTP calls)
# I/O-bound services benefit from multiple workers because while one worker
# waits for the encoder response, another can handle a new request.
# The encoder is CPU-bound — multiple workers there would fight for CPU.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]