#!/usr/bin/env python3
"""
RunPod LLaVA Image Processing Script
Processes artwork images and generates descriptions using LLaVA
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGES_DIR = "/workspace/images"
OUTPUT_DIR = "/workspace/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "llava_captions.jsonl")
MAX_NEW_TOKENS = 260
FLUSH_EVERY = 10

print("="*70)
print("LLAVA ARTWORK PROCESSOR")
print("="*70)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("\n[1/4] Loading LLaVA model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("✓ Model loaded successfully")

# Helper functions
def resize_for_vlm(img: Image.Image, max_side: int = 768) -> Image.Image:
    """Resize image for VLM processing"""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def llava_describe(image: Image.Image, max_new_tokens: int = 260) -> str:
    """Generate artwork description using LLaVA"""
    instruction = (
        "You are analysing an artwork image.\n"
        "Step 1: Give a short objective description (what is visibly present).\n"
        "Step 2: Provide a conceptual interpretation (possible meaning, symbolism, intent).\n"
        "Step 3: List 3–6 visual cues that justify your interpretation.\n"
        "Step 4: Offer up to 2 alternative interpretations if plausible.\n"
        "\n"
        "IMPORTANT:\n"
        "If there is any clearly readable text, words, or typography visible in the image,\n"
        "treat this text as part of the artwork itself and explicitly consider its meaning\n"
        "and its relationship to the visual elements in your interpretation.\n"
        "Do not invent or guess unreadable text.\n"
        "\n"
        "Rules: Do not guess the artist or title. Do not invent facts not supported by the image.\n"
        "Output as JSON-like text with keys:\n"
        "objective, interpretation, visual_cues, alternatives, confidence.\n"
    )

    prompt = "<image>\n" + instruction
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )

    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    # Remove prompt echo if present
    if instruction in text:
        text = text.split(instruction, 1)[-1].strip()

    return text


# Find images
print("\n[2/4] Scanning for images...")
img_exts = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")

if not os.path.isdir(IMAGES_DIR):
    print(f"ERROR: Images directory not found: {IMAGES_DIR}")
    print("Please upload your images to /workspace/images/")
    exit(1)

# Load checkpoint
done = set()
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if record.get("status") == "ok":
                        done.add(record["image_filename"])
        print(f"✓ Checkpoint loaded: {len(done)} images already processed")
    except Exception as e:
        print(f"⚠ Could not load checkpoint: {e}")
        done = set()

files = sorted([fn for fn in os.listdir(IMAGES_DIR) if fn.lower().endswith(img_exts)])
todo = [fn for fn in files if fn not in done]

print(f"\n{'='*70}")
print(f"PROCESSING SUMMARY")
print(f"{'='*70}")
print(f"Total images found : {len(files)}")
print(f"Already processed  : {len(done)}")
print(f"To process now     : {len(todo)}")
print(f"Output file        : {OUTPUT_FILE}")
print(f"{'='*70}\n")

if len(todo) == 0:
    print("✓ All images already processed!")
    exit(0)

# Process images
print("[3/4] Processing images...")
rows_buffer = []
start_time = time.time()

for idx, fn in enumerate(tqdm(todo, desc="Processing"), 1):
    path = os.path.join(IMAGES_DIR, fn)
    t0 = time.time()

    try:
        img = Image.open(path).convert("RGB")
        img = resize_for_vlm(img, max_side=768)

        desc = llava_describe(img, max_new_tokens=MAX_NEW_TOKENS)
        desc = desc.replace("\r\n", "\n").strip()

        elapsed = round(time.time() - t0, 2)

        record = {
            "image_filename": fn,
            "llava_description": desc,
            "processing_time_seconds": elapsed,
            "status": "ok",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embeddings": None,
            "web_metadata": None,
            "similarity_scores": None
        }

        rows_buffer.append(record)

        # Progress update every 10 images
        if idx % 10 == 0:
            avg_time = (time.time() - start_time) / idx
            remaining = len(todo) - idx
            eta_seconds = remaining * avg_time
            eta_hours = eta_seconds / 3600
            print(f"\n✓ Processed {idx}/{len(todo)} | Avg: {avg_time:.1f}s/image | ETA: {eta_hours:.1f}h")

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        print(f"\n✗ Error processing {fn}: {type(e).__name__}: {str(e)[:100]}")

        rows_buffer.append({
            "image_filename": fn,
            "llava_description": "",
            "processing_time_seconds": elapsed,
            "status": f"error: {type(e).__name__}: {e}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embeddings": None,
            "web_metadata": None,
            "similarity_scores": None
        })

    # Flush buffer
    if len(rows_buffer) >= FLUSH_EVERY:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for record in rows_buffer:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        rows_buffer = []

# Final flush
if rows_buffer:
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for record in rows_buffer:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"[4/4] PROCESSING COMPLETE!")
print(f"{'='*70}")
print(f"Total time     : {total_time/3600:.2f} hours")
print(f"Images processed: {len(todo)}")
print(f"Output saved to : {OUTPUT_FILE}")
print(f"{'='*70}\n")
print("Download your results from /workspace/output/")
