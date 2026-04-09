"""
services/api/main.py
====================
WHY THIS IS A SEPARATE SERVICE FROM THE ENCODER:
  This service handles:
    - FAISS index (search logic)
    - Whisper (voice transcription)
    - Request routing
    - Feedback storage
    - Result reranking

  The encoder handles:
    - ONNX inference (heavy ML model)

  Separation means: if FAISS crashes, encoder keeps running.
  If encoder needs to be swapped for GPU, API logic doesn't change.
  They communicate over HTTP on the internal Docker network.

WHISPER FOR VOICE SEARCH:
  OpenAI Whisper is a speech-to-text model.
  We use the "tiny" variant (39MB):
    tiny:   39MB,  ~2s for 5s audio,  ~88% word accuracy
    base:   74MB,  ~3s for 5s audio,  ~91% word accuracy
    small:  244MB, ~6s for 5s audio,  ~94% word accuracy
    medium: 769MB, ~15s for 5s audio, ~96% word accuracy
    large:  1.5GB, ~30s for 5s audio, ~98% word accuracy

  We chose TINY because:
    - Search queries are short (3-10 words), not medical transcription
    - 88% accuracy on "dog running in park" is effectively 100%
    - 2 seconds latency vs 30 seconds for large is massive UX difference
    - 39MB vs 1.5GB — fits comfortably in our Docker container

  TRADEOFF: If user has strong accent or says complex phrases,
  tiny might mishear. For a demo/portfolio, fine. For production,
  add an option to select model size.

THE RERANKER:
  FAISS returns top-K results by vector distance.
  Distance is a good but imperfect proxy for relevance.
  The reranker applies additional signals:
    1. Feedback boost: if user previously liked an image, boost similar ones
    2. Diversity: don't return 10 photos from the same category
    3. Recency: optionally boost recently added images

  This is a LIGHTWEIGHT reranker — no neural network, just heuristics.
  A full cross-encoder reranker (like BERT) would be more accurate but
  adds 50-100ms latency. For search, perceived speed matters more than
  marginal accuracy improvements.
"""

import os
import io
import pickle
import logging
import time
import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import faiss
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [api] %(message)s")
log = logging.getLogger(__name__)

# ── Configuration from environment ───────────────────────────────────────────
# Using env vars (not hardcoded) so Docker Compose can configure them
ENCODER_URL = os.getenv("ENCODER_URL", "http://encoder:8001")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "embeddings")
IMAGES_DIR = os.getenv("IMAGES_DIR", "images")
DB_PATH = os.getenv("DB_PATH", "data/search.db")
NPROBE = int(os.getenv("FAISS_NPROBE", "10"))

# ── Global state ──────────────────────────────────────────────────────────────
faiss_index = None
metadata: list[dict] = []
whisper_model = None
db_conn: Optional[sqlite3.Connection] = None


# ── Lifespan (replaces @app.on_event, modern FastAPI pattern) ─────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all resources on startup, clean up on shutdown."""
    global faiss_index, metadata, whisper_model, db_conn

    # Load FAISS index
    index_path = os.path.join(EMBEDDINGS_DIR, "faiss.index")
    meta_path = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")

    if Path(index_path).exists():
        log.info(f"Loading FAISS index from {index_path}...")
        faiss_index = faiss.read_index(index_path)
        faiss_index.nprobe = NPROBE  # set search-time parameter
        log.info(f"  Index loaded: {faiss_index.ntotal} vectors")
    else:
        log.warning(f"No FAISS index at {index_path}. Run ingest.py first.")

    if Path(meta_path).exists():
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        log.info(f"  Metadata loaded: {len(metadata)} records")

    # Load Whisper (lazy — only if installed)
    try:
        import whisper
        log.info("Loading Whisper tiny model for voice search...")
        whisper_model = whisper.load_model("tiny")
        log.info("  Whisper ready.")
    except ImportError:
        log.warning("Whisper not installed. Voice search disabled. "
                    "Install with: pip install openai-whisper")
    except Exception as e:
        log.warning(f"Whisper load failed: {e}")

    # Setup SQLite for feedback + query logging
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _init_db(db_conn)
    log.info("Database ready.")

    log.info("API service ready.")
    yield  # ← app runs here

    # Cleanup on shutdown
    if db_conn:
        db_conn.close()


def _init_db(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT,
            query_type TEXT,   -- 'text', 'image', 'voice'
            result_count INTEGER,
            latency_ms REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            query_text TEXT,
            vote INTEGER NOT NULL,  -- +1 = thumbs up, -1 = thumbs down
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()


app = FastAPI(
    title="Visual Search API",
    description="Semantic image search powered by CLIP + FAISS + Whisper",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve image files statically
# This lets the React frontend load actual images
images_path = Path(IMAGES_DIR)
if images_path.exists():
    app.mount("/images", StaticFiles(directory=str(images_path)), name="images")


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class SearchResult(BaseModel):
    path: str          # relative path for frontend to construct URL
    url: str           # full URL to fetch the image
    category: str
    score: float       # similarity score 0-1 (higher = more similar)
    rank: int

class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    query_type: str
    total_found: int
    latency_ms: float
    encoder_latency_ms: float

class FeedbackRequest(BaseModel):
    image_path: str
    query: str
    vote: int  # +1 or -1

class StatsResponse(BaseModel):
    total_images: int
    total_queries: int
    index_type: str
    nprobe: int
    whisper_available: bool


# ── Core search logic ─────────────────────────────────────────────────────────
async def get_embedding_for_text(text: str) -> tuple[np.ndarray, float]:
    """Call encoder service to get text embedding."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{ENCODER_URL}/embed/text",
            json={"text": text},
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Encoder error: {resp.text}")
        data = resp.json()
        return np.array(data["embedding"], dtype=np.float32), data["latency_ms"]


async def get_embedding_for_image(image_bytes: bytes) -> tuple[np.ndarray, float]:
    """Call encoder service to get image embedding."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{ENCODER_URL}/embed/image/upload",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Encoder error: {resp.text}")
        data = resp.json()
        return np.array(data["embedding"], dtype=np.float32), data["latency_ms"]


def faiss_search(
    query_embedding: np.ndarray,
    k: int = 20,
) -> list[tuple[int, float]]:
    """
    Search FAISS index.
    Returns list of (metadata_index, distance) sorted by distance ascending.

    WHY k=20 when user wants top-10:
      We fetch 20 (2x) because the reranker may reorder them.
      Fetching more candidates = reranker has more to work with.
      This is called "over-fetching" — standard practice in two-stage retrieval.
    """
    if faiss_index is None:
        raise HTTPException(503, "FAISS index not loaded. Run ingest.py first.")

    # FAISS expects shape [1, 512] for single query
    query = query_embedding.reshape(1, -1)

    # D = distances, I = indices into metadata list
    D, I = faiss_index.search(query, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:  # -1 means FAISS couldn't find enough results
            continue
        results.append((int(idx), float(dist)))

    return results


def rerank(
    results: list[tuple[int, float]],
    query: str,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """
    Apply feedback signals to reorder FAISS results.

    WHAT RERANKING DOES:
      FAISS gives us [img1, img2, img3...] ordered by vector distance.
      But vector distance doesn't know:
        - Which images a USER has liked before
        - Whether we're showing too many similar images (diversity)

      The reranker adjusts scores based on this context.

    FEEDBACK BOOST:
      If user previously gave thumbs up to an image similar to the query,
      we boost its score slightly. Not a lot — we don't want to overfit
      to one user's preferences, but enough to personalize.

    DIVERSITY PENALTY:
      If we already have 3 images from the same category in top results,
      the 4th one gets a small penalty. Prevents showing 10 dog photos
      when searching "animals".

    WHY NOT A NEURAL RERANKER:
      Cross-encoder models (BERT-based) can rerank with 95%+ accuracy
      but add 50-200ms latency per result set.
      Our lightweight heuristic adds <1ms.
      For a portfolio project, the heuristic is the right call.
      For a production search engine serving 10k QPS, neural reranking
      on a GPU is the right call.
    """
    if db_conn is None or not results:
        return results[:top_k]

    # Get feedback data for these image paths
    relevant_paths = [metadata[idx]["path"] for idx, _ in results if idx < len(metadata)]
    placeholders = ",".join(["?"] * len(relevant_paths))
    cursor = db_conn.execute(
        f"SELECT image_path, SUM(vote) as score FROM feedback "
        f"WHERE image_path IN ({placeholders}) GROUP BY image_path",
        relevant_paths,
    )
    feedback_scores = {row[0]: row[1] for row in cursor.fetchall()}

    # Diversity tracking
    category_counts: dict[str, int] = {}

    adjusted = []
    for idx, dist in results:
        if idx >= len(metadata):
            continue
        record = metadata[idx]
        path = record["path"]
        category = record.get("category", "unknown")

        # Convert L2 distance to similarity score [0, 1]
        # L2 distance 0 = identical, grows as vectors diverge
        # We convert: similarity = 1 / (1 + distance)
        similarity = 1.0 / (1.0 + dist)

        # Apply feedback boost
        user_vote = feedback_scores.get(path, 0)
        if user_vote > 0:
            similarity *= 1.15  # 15% boost for liked images
        elif user_vote < 0:
            similarity *= 0.70  # 30% penalty for disliked images

        # Apply diversity penalty
        count_in_category = category_counts.get(category, 0)
        if count_in_category >= 3:
            similarity *= 0.90  # 10% penalty if category is already represented

        category_counts[category] = count_in_category + 1
        adjusted.append((idx, similarity))

    # Sort by adjusted similarity descending
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted[:top_k]


def build_response(
    ranked: list[tuple[int, float]],
    query: str,
    query_type: str,
    encoder_latency: float,
    total_latency: float,
) -> SearchResponse:
    """Build the final response from ranked results."""
    results = []
    for rank, (idx, score) in enumerate(ranked):
        if idx >= len(metadata):
            continue
        record = metadata[idx]
        path = record["path"]
        # Convert local filesystem path to URL the frontend can use
        # Docker volume mounts images at /images/ route
        relative = path.replace("\\", "/")
        # Extract everything after 'images/'
        parts = relative.split("images/")
        url_path = parts[-1] if len(parts) > 1 else os.path.basename(path)

        results.append(SearchResult(
            path=path,
            url=f"/images/{url_path}",
            category=record.get("category", "unknown"),
            score=round(min(score, 1.0), 4),
            rank=rank + 1,
        ))

    return SearchResponse(
        results=results,
        query=query,
        query_type=query_type,
        total_found=len(results),
        latency_ms=round(total_latency, 1),
        encoder_latency_ms=round(encoder_latency, 1),
    )


def log_query(query: str, query_type: str, result_count: int, latency_ms: float):
    """Store query in SQLite for analytics."""
    if db_conn:
        try:
            db_conn.execute(
                "INSERT INTO queries (query_text, query_type, result_count, latency_ms) VALUES (?,?,?,?)",
                (query, query_type, result_count, latency_ms),
            )
            db_conn.commit()
        except Exception as e:
            log.warning(f"Failed to log query: {e}")


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_loaded": faiss_index is not None,
        "image_count": faiss_index.ntotal if faiss_index else 0,
        "whisper_available": whisper_model is not None,
    }


@app.get("/stats", response_model=StatsResponse)
async def stats():
    total_queries = 0
    if db_conn:
        row = db_conn.execute("SELECT COUNT(*) FROM queries").fetchone()
        total_queries = row[0] if row else 0

    index_type = "none"
    if faiss_index:
        index_type = type(faiss_index).__name__

    return StatsResponse(
        total_images=faiss_index.ntotal if faiss_index else 0,
        total_queries=total_queries,
        index_type=index_type,
        nprobe=NPROBE,
        whisper_available=whisper_model is not None,
    )


@app.get("/search/text")
async def search_text(q: str, k: int = 10):
    """
    Text → image search.
    User types "dog running in park" → returns top-k matching images.
    """
    if not q.strip():
        raise HTTPException(400, "Query cannot be empty")

    t0 = time.perf_counter()
    embedding, encoder_ms = await get_embedding_for_text(q)
    raw_results = faiss_search(embedding, k=k * 2)
    ranked = rerank(raw_results, q, top_k=k)
    latency = (time.perf_counter() - t0) * 1000

    log_query(q, "text", len(ranked), latency)
    return build_response(ranked, q, "text", encoder_ms, latency)


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), k: int = 10):
    """
    Image → similar image search (reverse image search).
    User uploads a photo → returns visually similar images.
    """
    t0 = time.perf_counter()
    contents = await file.read()
    embedding, encoder_ms = await get_embedding_for_image(contents)
    raw_results = faiss_search(embedding, k=k * 2)
    ranked = rerank(raw_results, "image_query", top_k=k)
    latency = (time.perf_counter() - t0) * 1000

    log_query("image_upload", "image", len(ranked), latency)
    return build_response(ranked, "image_upload", "image", encoder_ms, latency)


@app.post("/search/voice")
async def search_voice(file: UploadFile = File(...), k: int = 10):
    """
    Voice → image search.
    User speaks "show me photos of mountains at sunset"
    → Whisper transcribes → CLIP searches → returns images.

    Flow:
      Audio file → Whisper tiny → transcribed text → same as /search/text
    """
    if whisper_model is None:
        raise HTTPException(503, "Voice search not available. Whisper not installed.")

    t0 = time.perf_counter()

    # Save audio to temp file (Whisper needs a file path, not bytes)
    import tempfile
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Whisper transcription
        # fp16=False because we're on CPU (FP16 is GPU-only)
        t_whisper = time.perf_counter()
        result = whisper_model.transcribe(tmp_path, fp16=False, language="en")
        whisper_ms = (time.perf_counter() - t_whisper) * 1000
        transcription = result["text"].strip()
        log.info(f"Voice transcription ({whisper_ms:.0f}ms): '{transcription}'")
    finally:
        os.unlink(tmp_path)  # clean up temp file

    if not transcription:
        raise HTTPException(400, "Could not transcribe audio")

    # Now treat it exactly like a text search
    embedding, encoder_ms = await get_embedding_for_text(transcription)
    raw_results = faiss_search(embedding, k=k * 2)
    ranked = rerank(raw_results, transcription, top_k=k)
    latency = (time.perf_counter() - t0) * 1000

    log_query(transcription, "voice", len(ranked), latency)

    response = build_response(ranked, transcription, "voice", encoder_ms, latency)
    # Add transcription to response so frontend can show "I heard: ..."
    return {**response.dict(), "transcription": transcription, "whisper_ms": round(whisper_ms, 1)}


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Store user feedback (thumbs up/down) for a search result.
    Used by the reranker to personalize future results.
    """
    if req.vote not in (-1, 1):
        raise HTTPException(400, "vote must be +1 or -1")
    if db_conn:
        db_conn.execute(
            "INSERT INTO feedback (image_path, query_text, vote) VALUES (?,?,?)",
            (req.image_path, req.query, req.vote),
        )
        db_conn.commit()
    return {"status": "ok"}


@app.get("/queries/recent")
async def recent_queries(limit: int = 20):
    """Return recent search queries for analytics."""
    if db_conn is None:
        return {"queries": []}
    rows = db_conn.execute(
        "SELECT query_text, query_type, result_count, latency_ms, timestamp "
        "FROM queries ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return {"queries": [
        {"query": r[0], "type": r[1], "results": r[2],
         "latency_ms": r[3], "timestamp": r[4]}
        for r in rows
    ]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")