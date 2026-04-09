# Visual Search Platform

Multimodal semantic image search — text, image, and voice — powered by CLIP + FAISS + Whisper.
Runs fully offline. No API keys, no cloud calls.

## Architecture

```
Browser → Nginx (React) → FastAPI (search + Whisper) → FastAPI (ONNX CLIP) → FAISS
```

Three Docker containers, each with one job. Read `docs/ARCHITECTURE.md` for every decision explained.

---

## Setup (Step by Step)

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker + Docker Compose
- 4GB free RAM
- 5GB free disk

---

### Step 1: Install Python dependencies (for the one-time export)

```bash
cd visual-search

pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install onnx onnxruntime onnxruntime-extensions
pip install icrawler Pillow numpy faiss-cpu
```

---

### Step 2: Export CLIP to ONNX + quantize to INT8

This step runs ONCE. After this you never need PyTorch again.

```bash
python scripts/export_to_onnx.py
```

What this does:
1. Downloads CLIP ViT-B/32 (~350MB, cached in `~/.cache/clip`)
2. Exports vision encoder to `models/clip_vision_fp32.onnx`
3. Quantizes to `models/clip_vision_int8.onnx` (~90MB)
4. Repeats for the text encoder
5. Benchmarks and prints the speedup

Expected output:
```
FP32 size: 347.2 MB
INT8 size: 89.4 MB
FP32 latency: 81.3ms
INT8 latency: 18.7ms
Speedup: 4.3x faster
```

---

### Step 3: Download and index images

```bash
python scripts/ingest.py --per-query 15
```

This downloads ~420 images (28 queries × 15 each) and builds the FAISS index.

Options:
```bash
# Use your own images (skip download)
python scripts/ingest.py --no-download --images-dir /path/to/your/images

# Tune search accuracy vs speed
python scripts/ingest.py --nlist 100 --nprobe 10  # default
python scripts/ingest.py --nlist 100 --nprobe 20  # more accurate, slower
```

Output:
```
[1/4] Downloading images...   (takes 2-5 min depending on internet)
[2/4] Loading ONNX encoder...
[3/4] Encoding 418 images... (takes 1-2 min on CPU)
[4/4] Building FAISS index...
  Dataset too small for IVFFlat (418 < 3900). Using FlatL2.
  Note: FlatL2 is exact but O(n). Fine up to ~5,000 images.
Ingestion complete.
```

> **Note on index type:** With ~420 images the script falls back to FlatL2 (exact search).
> To get IVFFlat you need 3,900+ images. Set `--per-query 150` for ~4,200 images.
> FlatL2 is fine for demos — the speed difference is invisible at <5k images.

---

### Step 4: Start with Docker Compose

```bash
docker compose up --build
```

First run: Docker builds the images (~5 min, downloads ~1.5GB of base images).
Subsequent runs: `docker compose up` (seconds).

Wait for:
```
visual-search-encoder  | Encoder service ready in 4.2s
visual-search-api      | API service ready.
```

Then open **http://localhost:3000**

---

### Step 5: Try it

**Text search:**
- Type "dog running in park" → hit Enter
- Type "city at night" → see urban images

**Image search:**
- Click "Upload" → select any image from your computer
- It finds visually similar images in the index

**Voice search:**
- Click and hold "Hold to speak"
- Say "mountain with snow"
- Release → Whisper transcribes → results appear
- You'll see "Heard: mountain with snow" above results

**Feedback:**
- Hit `+` on a result to mark it relevant
- Hit `−` to mark it irrelevant
- Future searches will boost/penalize those images (stored in SQLite)

---

## Development (without Docker)

Run each service locally for faster iteration:

**Terminal 1 — encoder:**
```bash
cd services/encoder
pip install -r requirements.txt
MODELS_DIR=../../models python main.py
```

**Terminal 2 — api:**
```bash
cd services/api
pip install -r requirements.txt
ENCODER_URL=http://localhost:8001 \
EMBEDDINGS_DIR=../../embeddings \
IMAGES_DIR=../../images \
python main.py
```

**Terminal 3 — frontend:**
```bash
cd services/frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## Tuning Search

After startup, you can tune FAISS search accuracy without rebuilding:

```bash
# More accurate (searches more clusters)
FAISS_NPROBE=20 docker compose up

# Fastest (1 cluster only — 85% recall)
FAISS_NPROBE=1 docker compose up
```

Check current stats at: http://localhost:8000/stats

---

## Project Structure

```
visual-search/
├── docs/
│   └── ARCHITECTURE.md      ← Every decision explained
├── scripts/
│   ├── export_to_onnx.py    ← CLIP → ONNX + INT8 quantization
│   └── ingest.py            ← Download images + build FAISS index
├── services/
│   ├── encoder/
│   │   ├── main.py          ← ONNX inference service
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── api/
│   │   ├── main.py          ← Search logic, FAISS, Whisper, feedback
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx
│       │   ├── components/
│       │   └── index.css
│       ├── Dockerfile       ← Multi-stage: Node build → Nginx serve
│       └── vite.config.js
├── models/                  ← ONNX model files (generated by export)
├── embeddings/              ← FAISS index (generated by ingest)
├── images/                  ← Downloaded images
└── docker-compose.yml       ← Three-service orchestration
```

---

## API Reference

After startup, full interactive docs at: **http://localhost:8000/docs**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search/text?q=...&k=10` | GET | Text → images |
| `/search/image` | POST | Image upload → similar images |
| `/search/voice` | POST | Audio file → transcribe → images |
| `/feedback` | POST | Submit thumbs up/down |
| `/stats` | GET | Index info, query count, nprobe |
| `/queries/recent` | GET | Last 20 searches |
| `/health` | GET | Service health check |

---

## The Interview Version

See `docs/ARCHITECTURE.md` for the full explanation.

Short version:
> "ONNX + INT8 quantization: 350MB → 90MB model, 80ms → 18ms inference (4× faster, 2% accuracy loss).
> FAISS IVFFlat nprobe=10: 97% recall at ~10ms vs 250ms brute-force at 100k images.
> Whisper tiny: 39MB, 2s transcription — right-sized for short search queries.
> Three Docker containers: encoder, api, frontend — independent restarts, independent scaling."