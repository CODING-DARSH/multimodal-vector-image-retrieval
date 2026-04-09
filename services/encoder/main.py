"""
services/encoder/main.py
========================
WHY A SEPARATE ENCODER SERVICE:
  The encoder (ONNX CLIP model) is the heaviest component:
    - ~90MB model file to load into RAM
    - Startup time: ~3 seconds to initialize ONNX Runtime session
    - CPU-intensive: uses all cores during inference

  If we put this inside the API service:
    1. Every API restart also restarts the encoder (3s downtime)
    2. Can't scale encoder independently (what if we add GPU later?)
    3. API crashes take down inference capability
    4. Can't swap the model without touching search logic

  As a SEPARATE SERVICE:
    - Encoder loads once, stays up
    - API restarts don't kill it
    - Swap ONNX → TensorRT (GPU) by changing ONE service
    - Can run on a different machine if needed

  The communication cost: one HTTP call per search query (~1ms on localhost)
  That's a fine tradeoff for the decoupling benefits.

WHY FASTAPI OVER FLASK:
  Flask: synchronous, one request at a time per worker
  FastAPI: async, handles multiple concurrent requests with one worker

  For an encoder service that does CPU-bound inference:
    - Both are fine for single requests
    - FastAPI's automatic OpenAPI docs at /docs is useful for debugging
    - Pydantic validation catches malformed inputs before they hit inference
    - Type hints make the code self-documenting
    - FastAPI is what real ML serving frameworks (Ray Serve, BentoML) use
"""

import os
import io
import base64
import time
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import clip
from PIL import Image
from torchvision import transforms

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [encoder] %(message)s")
log = logging.getLogger(__name__)

# ── CLIP image preprocessing ─────────────────────────────────────────────────
# Replicated from CLIP source — we don't need all of PyTorch, just this transform
PREPROCESS = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Visual Search Encoder",
    description="ONNX INT8 CLIP encoder — converts images and text to 512-dim vectors",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighter in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global encoder state ──────────────────────────────────────────────────────
# WHY GLOBAL STATE (not dependency injection):
#   ONNX InferenceSession is NOT thread-safe to CREATE, but IS thread-safe to RUN.
#   We create it once at startup and share it.
#   FastAPI's @app.on_event("startup") runs before any requests are served.

vision_session: Optional[ort.InferenceSession] = None
text_session: Optional[ort.InferenceSession] = None
vision_input_name: str = ""
text_input_name: str = ""
startup_time: float = 0.0


@app.on_event("startup")
async def load_models():
    global vision_session, text_session, vision_input_name, text_input_name, startup_time

    models_dir = os.getenv("MODELS_DIR", "models")
    vision_path = os.path.join(models_dir, "clip_vision_int8.onnx")
    text_path = os.path.join(models_dir, "clip_text_int8.onnx")

    # Session options: tune threading
    # intra_op = parallelism within a single operation (e.g. matrix multiply)
    # inter_op = parallelism between operations in the graph
    # For inference-only with small batches: max intra, min inter
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count()
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        log.info("CUDA GPU available — using GPU for inference")
    else:
        log.info("No CUDA GPU found — using CPU with INT8 optimizations")

    t0 = time.perf_counter()

    if Path(vision_path).exists():
        vision_session = ort.InferenceSession(vision_path, opts, providers=providers)
        vision_input_name = vision_session.get_inputs()[0].name
        log.info(f"Vision encoder loaded: {vision_path}")
    else:
        log.warning(f"Vision model not found at {vision_path}. Run export_to_onnx.py first.")

    if Path(text_path).exists():
        text_session = ort.InferenceSession(text_path, opts, providers=providers)
        text_input_name = text_session.get_inputs()[0].name
        log.info(f"Text encoder loaded: {text_path}")
    else:
        log.warning(f"Text model not found at {text_path}. Run export_to_onnx.py first.")

    startup_time = time.perf_counter() - t0
    log.info(f"Encoder service ready in {startup_time:.2f}s")


# ── Pydantic models (request/response schemas) ────────────────────────────────
class TextEmbedRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list[float]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    vision_loaded: bool
    text_loaded: bool
    startup_time_s: float


# ── Helper functions ──────────────────────────────────────────────────────────
def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize a vector. Makes cosine similarity == dot product."""
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-8)


def embed_image_array(arr: np.ndarray) -> tuple[list[float], float]:
    """Run vision encoder on a preprocessed image array."""
    t0 = time.perf_counter()
    output = vision_session.run(None, {vision_input_name: arr})
    emb = l2_normalize(output[0][0])
    latency_ms = (time.perf_counter() - t0) * 1000
    return emb.tolist(), latency_ms


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Docker health check + status."""
    return HealthResponse(
        status="ok",
        vision_loaded=vision_session is not None,
        text_loaded=text_session is not None,
        startup_time_s=round(startup_time, 2),
    )


@app.post("/embed/text", response_model=EmbeddingResponse)
async def embed_text(req: TextEmbedRequest):
    """
    Convert text query → 512-dim CLIP embedding.
    Called by the API service on every text search.
    """
    if text_session is None:
        raise HTTPException(503, "Text encoder not loaded")
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    t0 = time.perf_counter()

    # Tokenize: convert text string → integer token IDs
    # CLIP uses a BPE tokenizer with max length 77
    # We still need the clip library for tokenization (it's tiny, no PyTorch needed at runtime)
    import clip as clip_tokenizer
    tokens = clip_tokenizer.tokenize([req.text]).numpy()  # shape: [1, 77]

    output = text_session.run(None, {text_input_name: tokens})
    emb = l2_normalize(output[0][0])

    latency_ms = (time.perf_counter() - t0) * 1000
    return EmbeddingResponse(embedding=emb.tolist(), latency_ms=round(latency_ms, 2))


@app.post("/embed/image/upload", response_model=EmbeddingResponse)
async def embed_image_upload(file: UploadFile = File(...)):
    """
    Convert uploaded image → 512-dim CLIP embedding.
    Used for reverse image search (search by image instead of text).
    """
    if vision_session is None:
        raise HTTPException(503, "Vision encoder not loaded")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    tensor = PREPROCESS(img).unsqueeze(0).numpy()
    emb, latency_ms = embed_image_array(tensor)
    return EmbeddingResponse(embedding=emb, latency_ms=round(latency_ms, 2))


@app.post("/embed/image/base64", response_model=EmbeddingResponse)
async def embed_image_base64(data: dict):
    """
    Convert base64-encoded image → embedding.
    Alternative to file upload for frontend that already has base64 data.
    """
    if vision_session is None:
        raise HTTPException(503, "Vision encoder not loaded")

    try:
        img_data = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 image: {e}")

    tensor = PREPROCESS(img).unsqueeze(0).numpy()
    emb, latency_ms = embed_image_array(tensor)
    return EmbeddingResponse(embedding=emb, latency_ms=round(latency_ms, 2))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")