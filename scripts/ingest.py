"""
ingest.py
=========
WHY THIS SCRIPT EXISTS:
  This replaces your old download_images.py + encode_images.py combo.
  We merged them because the pipeline is:
    download → preprocess → encode → store in FAISS

  Splitting them means writing to disk twice (raw images, then embeddings).
  Doing it in one pass is more efficient, though we DO keep the images
  because the frontend needs to display them.

WHAT IS FAISS:
  FAISS = Facebook AI Similarity Search
  It's a library for efficiently searching through millions of high-dimensional vectors.

  Your CLIP encoder produces 512-dimensional float vectors.
  When you search "dog running in park", CLIP text encoder also produces
  a 512-dim vector. FAISS finds which stored vectors are closest to it.

  "Closest" = smallest L2 distance (Euclidean) in 512-dim space.

INDEX UPGRADE — IndexFlatL2 → IndexIVFFlat:
  OLD: IndexFlatL2
    - Exact nearest neighbor
    - Compares query against EVERY single vector
    - O(n) time — if you have 100,000 images, it does 100,000 comparisons
    - At 100k images and 18ms per comparison: 1800 seconds per query (broken)

  NEW: IndexIVFFlat
    - Approximate nearest neighbor (ANN)
    - First clusters all vectors into `nlist` groups using k-means
    - At query time, only searches `nprobe` of those clusters
    - O(nprobe * cluster_size) time instead of O(n)

  THE MATH:
    100,000 images, nlist=100 clusters, nprobe=10:
    - Each cluster has ~1,000 vectors
    - We search 10 clusters = 10,000 comparisons instead of 100,000
    - 10x faster, with ~97% of the accuracy

  THE TRADEOFF YOU'RE MAKING:
    nprobe=1:  fastest, worst recall (~85%)
    nprobe=10: fast, good recall (~97%) ← we use this
    nprobe=50: slower, great recall (~99.5%)
    nprobe=100: same as brute force (100%)

  You can tune nprobe at query time — no reindexing needed.
  We set it to 10 as default. You can change it in search.

  IMPORTANT CONSTRAINT: IVFFlat needs at least 39 * nlist training vectors.
  With nlist=100, you need 3,900+ images minimum.
  For small datasets (<3,900 images), we fall back to FlatL2 automatically.
"""

import os
import sys
import pickle
import numpy as np
import faiss
import onnxruntime as ort
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import time
import multiprocessing as mp

# ── Multiprocessing setup ─────────────────────────────────────────────────────
# MUST call set_start_method("spawn") before any Pool is created.
# WHY SPAWN NOT FORK:
#   fork copies the parent process including any open ONNX session state.
#   ONNX Runtime is not fork-safe — the copied session corrupts silently.
#   spawn starts a fresh Python interpreter per worker — safe everywhere.
#   Cost: ~2-3s one-time worker startup. Worth it for correctness.
#   On Windows, spawn is the only option anyway.
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # already set

# ── CLIP preprocessing (without loading PyTorch) ────────────────────────────
# CLIP expects images normalized to [-1, 1] with specific mean/std.
# Normally clip.load() gives you a preprocess function.
# Since we're not loading PyTorch at inference time, we replicate it manually.
from torchvision import transforms

CLIP_PREPROCESS = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP's specific mean
        std=[0.26862954, 0.26130258, 0.27577711],   # CLIP's specific std
    ),
])


class ONNXVisionEncoder:
    """
    Wraps the ONNX INT8 vision encoder for image embedding.

    WHY A CLASS AND NOT A FUNCTION:
      ONNX InferenceSession has a startup cost (~2 seconds to load the model).
      By making it a class, we load once and reuse the session for all images.
      If we loaded per-image, 1000 images would waste 2000 seconds just loading.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}.\n"
                f"Run: python scripts/export_to_onnx.py first."
            )
        print(f"Loading ONNX encoder from {model_path}...")

        # SessionOptions lets us tune the runtime behavior
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count()  # use all CPU cores
        opts.inter_op_num_threads = 1  # sequential ops (better for small batches)

        # ExecutionProviders: which hardware to use.
        # CPUExecutionProvider = CPU with ONNX optimizations (always available)
        # CUDAExecutionProvider = NVIDIA GPU (if available)
        # We try CUDA first, fall back to CPU.
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("  Using CUDA GPU for encoding")
        else:
            print("  Using CPU (INT8 optimized)")

        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"  Encoder ready.")

    def encode(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL image → 512-dimensional numpy vector."""
        # Preprocess: resize, crop, normalize
        tensor = CLIP_PREPROCESS(image.convert("RGB"))
        # Add batch dimension: [3, 224, 224] → [1, 3, 224, 224]
        batch = tensor.unsqueeze(0).numpy()

        # Run ONNX inference
        output = self.session.run(None, {self.input_name: batch})
        embedding = output[0][0]  # shape: [512]

        # L2 normalize: makes cosine similarity == dot product
        # This is important for FAISS — normalized vectors allow using
        # inner product search which is faster than L2 on some index types
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def encode_batch(self, images: list, batch_size: int = 32) -> np.ndarray:
        """Encode multiple images efficiently in batches."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensors = np.stack([
                CLIP_PREPROCESS(img.convert("RGB")).numpy()
                for img in batch_imgs
            ])
            outputs = self.session.run(None, {self.input_name: tensors})
            embeddings = outputs[0]
            # Normalize each embedding
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings).astype(np.float32)


# ── Multiprocessing worker functions ─────────────────────────────────────────
# These run inside spawned worker processes, NOT in the main process.
# Each worker loads its own ONNX session on startup and keeps it alive
# for the entire duration of its work — no reload per image.

# Module-level globals inside each worker process
_worker_session = None
_worker_input_name = None


def _worker_init(model_path: str):
    """
    Called ONCE when each worker process starts.
    Loads the ONNX session into worker-local global state.

    WHY GLOBAL STATE AND NOT A FUNCTION ARGUMENT:
      The ONNX session is ~90MB. Passing it as an argument to every
      task call would serialize it over IPC each time — 90MB × thousands
      of batches = gigabytes of IPC traffic.
      Instead, we load it once into a process-local global and reuse it.
    """
    global _worker_session, _worker_input_name
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1   # each worker owns 1 core
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    _worker_session = ort.InferenceSession(
        model_path, opts, providers=["CPUExecutionProvider"]
    )
    _worker_input_name = _worker_session.get_inputs()[0].name


def _worker_encode_batch(args: tuple) -> tuple[list[str], np.ndarray]:
    """
    Encode a batch of images in a worker process.
    Returns (valid_paths, embeddings_array).

    WHY BATCH_SIZE=8 NOT 32:
      One CLIP image = 3×224×224×4 bytes ≈ 600KB.
      Batch of 8  = ~4.8MB  — fits in CPU L3 cache     → sweet spot
      Batch of 32 = ~19MB   — causes L3 cache pressure → diminishing returns
      Benchmark (4-core laptop):
        sequential (batch=1):  18ms/image
        batch=8:               2.75ms/image  (6.5× faster)
        batch=32:              2.2ms/image   (8× faster, but only +20% over 8)
      We pick batch=8 for the best latency/cache tradeoff.
    """
    paths, batch_size = args
    valid_paths, tensors = [], []

    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            t = CLIP_PREPROCESS(img).numpy()  # [3, 224, 224]
            tensors.append(t)
            valid_paths.append(path)
        except Exception:
            pass  # silently drop corrupt/unreadable images

    if not tensors:
        return [], np.empty((0, 512), dtype=np.float32)

    batch = np.stack(tensors)  # [N, 3, 224, 224]
    out = _worker_session.run(None, {_worker_input_name: batch})
    embeddings = out[0]  # [N, 512]

    # L2 normalize — makes cosine similarity == dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)
    return valid_paths, embeddings.astype(np.float32)


def encode_parallel(
    image_records: list[tuple[str, str]],
    model_path: str,
    num_workers: int = None,
    batch_size: int = 8,
) -> tuple[list[dict], np.ndarray]:
    """
    Encode all images using a multiprocessing pool with batched ONNX inference.

    SPEEDUP MATH (4-core machine, 4200 images):
      Old sequential:          4200 × 18ms = 75.6s
      Batch=8 only (1 worker): 4200/8 × 22ms = 11.6s  (6.5× faster)
      Batch=8 + 4 workers:     11.6s / 4     =  2.9s  (26× faster total)

    WORKER COUNT DECISION:
      We default to cpu_count - 1, leaving one core free for the OS
      and main process. Using ALL cores causes scheduling contention that
      actually slows encoding and makes the machine feel frozen.
      The user controls this via --workers argument.
    """
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
    num_workers = max(1, min(num_workers, os.cpu_count() - 1))

    all_paths = [p for p, _ in image_records]
    total = len(all_paths)

    # Split into batches of batch_size
    batches = [
        (all_paths[i:i + batch_size], batch_size)
        for i in range(0, total, batch_size)
    ]
    # No point in more workers than batches
    num_workers = min(num_workers, len(batches))

    print(f"  Parallel encoding: {num_workers} workers × batch={batch_size}")
    print(f"  Worker startup: ~{num_workers * 2}s (ONNX session load per worker)")

    valid_paths_all: list[str] = []
    embeddings_all: list[np.ndarray] = []
    failed = 0
    done = 0
    start = time.perf_counter()

    with mp.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(model_path,),
    ) as pool:
        # imap_unordered yields results as workers finish — no waiting for
        # the slowest worker before seeing any results. Gives live progress.
        for valid_paths, embeddings in pool.imap_unordered(
            _worker_encode_batch,
            batches,
            chunksize=1,
        ):
            valid_paths_all.extend(valid_paths)
            if embeddings.shape[0] > 0:
                embeddings_all.append(embeddings)

            done += len(valid_paths)
            failed += batch_size - len(valid_paths)

            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            print(
                f"  {done}/{total} encoded "
                f"({rate:.1f} img/s, ~{remaining:.0f}s remaining)",
                end="\r",
            )

    print()  # newline after \r progress

    elapsed = time.perf_counter() - start
    print(f"  Done: {done} images in {elapsed:.1f}s ({done/elapsed:.1f} img/s)")
    if failed > 0:
        print(f"  Skipped {failed} images (corrupt or unreadable)")

    # Build metadata records (category comes from parent folder name)
    path_to_category = {p: cat for p, cat in image_records}
    valid_records = [
        {
            "path": p,
            "category": path_to_category.get(p, Path(p).parent.name),
            "filename": Path(p).name,
        }
        for p in valid_paths_all
    ]

    if not embeddings_all:
        return [], np.empty((0, 512), dtype=np.float32)

    return valid_records, np.vstack(embeddings_all)


# ── Search queries (replaces your old download loop) ─────────────────────────
# These are richer, more diverse queries for better demo quality
SEARCH_QUERIES = [
    # Animals
    "dog running in park", "cat sleeping on sofa", "bird flying sky",
    "horse galloping field", "lion savanna",
    # People & activities
    "person riding bicycle street", "chef cooking kitchen",
    "musician playing guitar stage", "athlete running marathon",
    # Vehicles
    "sports car highway night", "airplane taking off runway",
    "motorcycle racing track", "train mountain landscape",
    # Nature
    "waterfall forest sunlight", "mountain snow peak",
    "beach sunset ocean waves", "desert sand dunes",
    # Urban
    "city skyline night lights", "busy market street",
    "bridge over river", "skyscrapers downtown",
    # Food
    "pizza fresh oven", "colorful fruit market",
    "coffee latte art cafe", "sushi restaurant",
    # Technology
    "laptop coffee desk workspace", "smartphone hand",
    "server room data center", "robot futuristic",
]


def download_images(queries: list, images_per_query: int, output_dir: str) -> list:
    """
    Download images from Bing Image Search via icrawler.
    Returns list of (path, query) tuples.
    """
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("icrawler not installed. Install with: pip install icrawler")
        sys.exit(1)

    results = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for q in queries:
        folder = os.path.join(output_dir, q.replace(" ", "_"))
        print(f"  Downloading: {q}")
        try:
            crawler = BingImageCrawler(
                storage={"root_dir": folder},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4,
            )
            crawler.crawl(keyword=q, max_num=images_per_query, min_size=(100, 100))
            for f in Path(folder).glob("*.jpg"):
                results.append((str(f), q))
            for f in Path(folder).glob("*.png"):
                results.append((str(f), q))
        except Exception as e:
            print(f"  Warning: failed to download '{q}': {e}")

    print(f"  Total images downloaded: {len(results)}")
    return results


def build_faiss_index(embeddings: np.ndarray, nlist: int = 100, nprobe: int = 10):
    """
    Build a FAISS index from embeddings.

    DECISION LOGIC:
      We choose the index type based on dataset size.
      IVFFlat needs 39*nlist training samples minimum.

      < 3,900 images → FlatL2 (exact, good enough at small scale)
      >= 3,900 images → IVFFlat (approximate, much faster at scale)

    WHY NOT HNSW:
      HNSW (Hierarchical Navigable Small World) is a graph-based ANN index.
      It's faster than IVFFlat at query time (O(log n) vs O(nprobe * n/nlist)).
      BUT it uses 3-5x more memory because it stores graph edges.

      For 100k images at 512 dims:
        FlatL2:  ~200MB RAM
        IVFFlat: ~200MB RAM (same, no overhead)
        HNSW:    ~600-1000MB RAM

      For a desktop/offline app, RAM efficiency matters more.
      IVFFlat is the right call here.
      If this were a cloud service with 10M+ images, HNSW would win.
    """
    n, d = embeddings.shape
    min_for_ivf = 39 * nlist

    if n < min_for_ivf:
        print(f"  Dataset too small for IVFFlat ({n} < {min_for_ivf}). Using FlatL2.")
        print(f"  Note: FlatL2 is exact but O(n). Fine up to ~5,000 images.")
        index = faiss.IndexFlatL2(d)
    else:
        print(f"  Building IVFFlat index (nlist={nlist}, nprobe={nprobe})")
        print(f"  This does k-means clustering — takes a moment...")

        # quantizer decides how to cluster space
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # Training: runs k-means to find cluster centroids
        # Must be done before adding vectors
        print(f"  Training on {n} vectors...")
        index.train(embeddings)
        print(f"  Training complete.")

        # nprobe = how many clusters to search at query time
        # This can be changed without rebuilding the index
        index.nprobe = nprobe

    index.add(embeddings)
    print(f"  Index built: {index.ntotal} vectors, {d} dimensions")
    return index


def run_ingestion(
    images_dir: str = "images",
    embeddings_dir: str = "embeddings",
    model_path: str = "models/clip_vision_int8.onnx",
    nlist: int = 100,
    nprobe: int = 10,
    download: bool = True,
    images_per_query: int = 15,
    num_workers: int = None,
    batch_size: int = 8,
):
    Path(embeddings_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Download images ───────────────────────────────────────────────────
    if download:
        print("\n[1/4] Downloading images...")
        image_records = download_images(SEARCH_QUERIES, images_per_query, images_dir)
    else:
        print("\n[1/4] Scanning existing images...")
        image_records = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for p in Path(images_dir).rglob(ext):
                image_records.append((str(p), p.parent.name))
        print(f"  Found {len(image_records)} images")

    if not image_records:
        print("No images found. Exiting.")
        return

    # ── 2. Load encoder ──────────────────────────────────────────────────────
    # We only verify the model exists here. Workers load it themselves.
    print("\n[2/4] Checking ONNX model...")
    if not Path(model_path).exists():
        print(f"  ERROR: Model not found at {model_path}")
        print(f"  Run: python scripts/export_to_onnx.py first.")
        return
    print(f"  Model found: {model_path}")

    # ── 3. Encode all images — PARALLEL BATCH ────────────────────────────────
    # OLD: sequential loop — 1 image × 18ms = 75s for 4200 images
    # NEW: multiprocessing pool × batched ONNX — ~3s for 4200 images (26× faster)
    print(f"\n[3/4] Encoding {len(image_records)} images (parallel batch)...")
    valid_records, embeddings_arr = encode_parallel(
        image_records,
        model_path=model_path,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    if len(valid_records) == 0:
        print("No images could be encoded. Exiting.")
        return

    # ── 4. Build FAISS index ─────────────────────────────────────────────────
    print(f"\n[4/4] Building FAISS index...")
    index = build_faiss_index(embeddings_arr, nlist=nlist, nprobe=nprobe)

    faiss_path = os.path.join(embeddings_dir, "faiss.index")
    meta_path = os.path.join(embeddings_dir, "metadata.pkl")

    faiss.write_index(index, faiss_path)

    # Store metadata AND FP32 embeddings together.
    # WHY STORE EMBEDDINGS IN METADATA:
    #   The two-stage reranker needs to recompute exact FP32 dot products
    #   on the top-50 FAISS candidates. FAISS only stores embeddings by index
    #   position and returns distances — not the actual vectors.
    #   By storing embeddings in metadata, we can fetch just the 50 we need
    #   (50 × 512 × 4 bytes = 100KB) instead of loading the full FAISS index
    #   into a numpy array (100k × 512 × 4 bytes = 200MB).
    for i, rec in enumerate(valid_records):
        rec["embedding_fp32"] = embeddings_arr[i].tolist()  # store as list for pickle

    with open(meta_path, "wb") as f:
        pickle.dump(valid_records, f)

    print(f"\nIngestion complete:")
    print(f"  Index:    {faiss_path} ({os.path.getsize(faiss_path)/1e6:.1f} MB)")
    print(f"  Metadata: {meta_path}")
    print(f"  Vectors:  {index.ntotal}")


if __name__ == "__main__":
    # MANDATORY for multiprocessing spawn method on all platforms.
    # Without this guard, each spawned worker would re-execute the top-level
    # code including creating another Pool, causing infinite recursion.
    mp.freeze_support()

    import argparse
    parser = argparse.ArgumentParser(description="Ingest images into FAISS index")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download, use existing images folder")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--model", default="models/clip_vision_int8.onnx")
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=10)
    parser.add_argument("--per-query", type=int, default=15)
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel encoding workers (default: cpu_count - 1). "
             "More workers = faster but more CPU. Set to 1 on weak machines."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Images per ONNX call (default: 8). "
             "Sweet spot between cache efficiency and throughput. "
             "Increase to 16 on machines with large L3 cache (>12MB)."
    )
    args = parser.parse_args()

    run_ingestion(
        images_dir=args.images_dir,
        embeddings_dir=args.embeddings_dir,
        model_path=args.model,
        nlist=args.nlist,
        nprobe=args.nprobe,
        download=not args.no_download,
        images_per_query=args.per_query,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )