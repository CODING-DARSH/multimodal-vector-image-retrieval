"""
export_to_onnx.py
=================
WHY THIS FILE EXISTS:
  The standard `clip` Python package loads a full PyTorch model at ~350MB.
  Every time you run inference (embed an image), PyTorch has to:
    1. Load its entire runtime
    2. Run the computation graph dynamically (it builds as it runs)
    3. Do everything in FP32 (32-bit floats)

  ONNX (Open Neural Network Exchange) is a frozen, static computation graph.
  Once exported, ONNX Runtime can:
    1. Optimize the graph (fuse operations, remove redundant nodes)
    2. Run it on ANY hardware with the same file (CPU, GPU, ARM, etc.)
    3. Apply quantization — converting weights from FP32 to INT8

WHAT IS INT8 QUANTIZATION:
  FP32 means every weight in the model is stored as a 32-bit float.
  INT8 means we convert them to 8-bit integers.

  How? For each layer we compute:
    scale = max(|weights|) / 127
    quantized_weight = round(weight / scale)

  Example:
    FP32 weight: 0.823 (stored as 4 bytes)
    INT8 weight: 104   (stored as 1 byte)

  WHY DOES THIS MAKE IT FASTER?
    1. MODEL SIZE: 4 bytes → 1 byte per weight = ~4x smaller (350MB → ~90MB)
    2. MEMORY BANDWIDTH: CPU spends less time fetching weights from RAM
    3. SIMD INSTRUCTIONS: Modern CPUs have INT8 SIMD that process 4x more
       values per clock cycle than FP32 SIMD
    4. CACHE HITS: Smaller model fits better in L2/L3 CPU cache

  ACCURACY TRADEOFF:
    INT8 has range [-127, 127] vs FP32's massive range.
    We lose some precision. In practice for CLIP ViT-B/32:
    - FP32 recall@10: ~100% (exact)
    - INT8 recall@10: ~97-98% (approximate)
    We trade 2-3% accuracy for 4x speed. Worth it for a search engine.

WHY ONLY THE VISION ENCODER, NOT THE TEXT ENCODER?
  CLIP has two parts:
    - Vision encoder (ViT): encodes images → 512-dim vector
    - Text encoder (Transformer): encodes text queries → 512-dim vector

  At INDEX TIME: we run vision encoder once per image (offline, slow is ok)
  At SEARCH TIME: we run text encoder once per query (real-time, must be fast)

  We quantize BOTH, but the vision encoder matters more because:
    - It's heavier (more conv layers)
    - It runs during indexing which could be thousands of images

  The text encoder is already small and fast.
"""

import os
import sys
import numpy as np
import torch
import clip
from pathlib import Path

# onnxruntime.quantization is the quantization toolkit
# It works AFTER export — we export FP32 first, quantize second
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_clip_vision_encoder(output_dir: str = "models"):
    """
    Step 1: Export CLIP vision encoder from PyTorch → ONNX (FP32)
    Step 2: Quantize ONNX FP32 → ONNX INT8
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fp32_path = os.path.join(output_dir, "clip_vision_fp32.onnx")
    int8_path = os.path.join(output_dir, "clip_vision_int8.onnx")

    print("Loading CLIP ViT-B/32 in PyTorch...")
    # We need PyTorch ONLY for the export step.
    # After this, we never load PyTorch again for inference.
    device = "cpu"  # Export must happen on CPU for ONNX compatibility
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # ── STEP 1: Export to ONNX (FP32) ──────────────────────────────────────
    print("Exporting vision encoder to ONNX (FP32)...")

    # We only want the VISION encoder half of CLIP.
    # CLIP's encode_image() is the vision transformer.
    # We wrap it in a thin module so torch.onnx.export knows the interface.
    class VisionEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.visual = clip_model.visual

        def forward(self, image):
            # CLIP internally normalizes to float before the visual encoder
            return self.visual(image.to(torch.float32))

    vision_model = VisionEncoder(model)
    vision_model.eval()

    # Dummy input: batch=1, channels=3, height=224, width=224
    # CLIP ViT-B/32 expects 224x224 RGB images
    dummy_input = torch.randn(1, 3, 224, 224)

    # torch.onnx.export traces the computation graph.
    # opset_version=14 is the ONNX operator set version.
    # Higher = more operators available, better optimization.
    # 14 is stable and supported by onnxruntime >= 1.14
    torch.onnx.export(
        vision_model,
        dummy_input,
        fp32_path,
        opset_version=14,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image": {0: "batch_size"},       # batch size can vary
            "embedding": {0: "batch_size"},
        },
        do_constant_folding=True,  # fuse constant operations at export time
    )
    print(f"  FP32 model saved: {fp32_path}")
    print(f"  FP32 size: {os.path.getsize(fp32_path) / 1e6:.1f} MB")

    # ── STEP 2: Quantize FP32 → INT8 ────────────────────────────────────────
    print("Quantizing to INT8...")

    # quantize_dynamic = dynamic quantization
    #
    # DYNAMIC vs STATIC quantization — important distinction:
    #
    # STATIC quantization:
    #   Requires a "calibration dataset" — you run 100+ real images through
    #   the model first to measure the actual range of activations, THEN
    #   quantize. More accurate but needs extra setup.
    #
    # DYNAMIC quantization:
    #   Quantizes weights at conversion time.
    #   Activations (intermediate values) are quantized at runtime, on the fly.
    #   No calibration data needed.
    #   Slightly less accurate than static, but the difference for CLIP is
    #   negligible (~0.5% recall difference) and setup is much simpler.
    #
    # For a search engine demo, dynamic is the correct choice.
    # For production medical imaging, you'd use static with a calibration set.

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,  # QInt8 = signed 8-bit integer [-127, 127]
        # QUInt8 = unsigned [0, 255] — used for activations after ReLU
        # We use QInt8 for weights since weights can be negative
    )

    int8_size = os.path.getsize(int8_path) / 1e6
    fp32_size = os.path.getsize(fp32_path) / 1e6
    print(f"  INT8 model saved: {int8_path}")
    print(f"  INT8 size: {int8_size:.1f} MB")
    print(f"  Size reduction: {fp32_size/int8_size:.1f}x smaller")

    # ── STEP 3: Benchmark ────────────────────────────────────────────────────
    print("\nBenchmarking inference speed...")

    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # FP32 benchmark
    sess_fp32 = onnxruntime.InferenceSession(fp32_path)
    import time
    times = []
    for _ in range(20):
        t = time.perf_counter()
        sess_fp32.run(None, {"image": test_input})
        times.append(time.perf_counter() - t)
    fp32_ms = np.mean(times[5:]) * 1000  # skip warmup
    print(f"  FP32 latency: {fp32_ms:.1f}ms")

    # INT8 benchmark
    sess_int8 = onnxruntime.InferenceSession(int8_path)
    times = []
    for _ in range(20):
        t = time.perf_counter()
        sess_int8.run(None, {"image": test_input})
        times.append(time.perf_counter() - t)
    int8_ms = np.mean(times[5:]) * 1000
    print(f"  INT8 latency: {int8_ms:.1f}ms")
    print(f"  Speedup: {fp32_ms/int8_ms:.1f}x faster")

    print(f"\nExport complete. Use {int8_path} for the encoder service.")
    return int8_path


def export_clip_text_encoder(output_dir: str = "models"):
    """
    Export the TEXT encoder half of CLIP.
    Used at query time: user types text → text encoder → 512d vector → FAISS search
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fp32_path = os.path.join(output_dir, "clip_text_fp32.onnx")
    int8_path = os.path.join(output_dir, "clip_text_int8.onnx")

    print("Loading CLIP for text encoder export...")
    model, _ = clip.load("ViT-B/32", device="cpu")
    model.eval()

    class TextEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.model = clip_model

        def forward(self, text_tokens):
            # text_tokens: tokenized text, shape [batch, 77]
            # 77 = CLIP's max context length
            return self.model.encode_text(text_tokens)

    text_model = TextEncoder(model)
    text_model.eval()

    # Dummy tokenized text input
    dummy_text = clip.tokenize(["a photo of a cat"])  # shape [1, 77]

    torch.onnx.export(
        text_model,
        dummy_text,
        fp32_path,
        opset_version=14,
        input_names=["text_tokens"],
        output_names=["embedding"],
        dynamic_axes={
            "text_tokens": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    print(f"  Text FP32 model: {fp32_path} ({os.path.getsize(fp32_path)/1e6:.1f} MB)")

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    print(f"  Text INT8 model: {int8_path} ({os.path.getsize(int8_path)/1e6:.1f} MB)")

    return int8_path


if __name__ == "__main__":
    print("=" * 60)
    print("CLIP → ONNX Export + INT8 Quantization")
    print("=" * 60)
    export_clip_vision_encoder("models")
    export_clip_text_encoder("models")
    print("\nDone. Both encoders are ready for the encoder service.")