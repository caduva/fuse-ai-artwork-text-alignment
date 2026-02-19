#!/usr/bin/env python3
"""
RunPod LLaVA Image Processing Script
Processes artwork images and generates structured descriptions using LLaVA.

Outputs:
- JSONL: /workspace/output/llava_captions.jsonl  (one JSON object per line)
- JSON : /workspace/output/llava_captions.json   (array JSON for JSONCrack)

Key change vs previous version:
- llava_description is saved as a real JSON object (dict), not a string.
- If parsing fails, we store raw_output + parse_error, without breaking JSONL.
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# =========================
# Configuration
# =========================
MODEL_ID = os.getenv("MODEL_ID", "llava-hf/llava-1.5-7b-hf")

IMAGES_DIR = os.getenv("IMAGES_DIR", "/workspace/images")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/workspace/output")

OUTPUT_FILE_JSONL = os.path.join(OUTPUT_DIR, "llava_captions.jsonl")
OUTPUT_FILE_JSON = os.path.join(OUTPUT_DIR, "llava_captions.json")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "260"))
FLUSH_EVERY = int(os.getenv("FLUSH_EVERY", "10"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "768"))

# If true, we re-run only the model output repair step when JSON parsing fails
ENABLE_REPAIR_ATTEMPT = os.getenv("ENABLE_REPAIR_ATTEMPT", "1").strip() not in ("0", "false", "False")

# Print header
print("=" * 70)
print("LLAVA ARTWORK PROCESSOR (STRUCTURED JSON OUTPUT)")
print("=" * 70)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load model
# =========================
print("\n[1/4] Loading LLaVA model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print("✓ Model loaded successfully")
print(f"  MODEL_ID: {MODEL_ID}")
print(f"  DEVICE  : {model.device}")

# =========================
# Prompt
# =========================
INSTRUCTION = (
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
    "Return valid JSON only (no extra commentary, no markdown). Use these keys exactly:\n"
    "objective, interpretation, visual_cues, alternatives, confidence.\n"
)

# =========================
# Helper functions
# =========================
def resize_for_vlm(img: Image.Image, max_side: int = 768) -> Image.Image:
    """Resize image for VLM processing."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def _extract_first_json_obj(text: str) -> Optional[str]:
    """
    Extract the first JSON object substring from a model output.
    This handles cases where the model echoes some text before/after.
    """
    t = (text or "").strip()

    # Normalize common artifact: model outputs 'visual\_cues' (escaped underscore)
    t = t.replace("\\_", "_")

    # If the model echoed the instruction, remove it
    if INSTRUCTION in t:
        t = t.split(INSTRUCTION, 1)[-1].strip()

    # Find first {...} block
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    return m.group(0).strip()


def _safe_json_loads(s: str) -> Dict[str, Any]:
    """
    Parse JSON string. If it fails, raise.
    """
    return json.loads(s)


def _repair_to_json(raw_text: str) -> Dict[str, Any]:
    """
    Attempt to coerce a malformed JSON-like output into valid JSON using the language model itself.
    This is text-only (no image) and cheap relative to re-running vision.
    """
    repair_prompt = (
        "Convert the following text into VALID JSON ONLY.\n"
        "Rules:\n"
        "- Output must be a single JSON object.\n"
        "- Use keys exactly: objective, interpretation, visual_cues, alternatives, confidence.\n"
        "- visual_cues must be an array of strings.\n"
        "- alternatives must be an array of strings.\n"
        "- If a field is missing, use empty string or empty array accordingly.\n"
        "- Do NOT include any extra keys.\n"
        "- Do NOT include any markdown.\n\n"
        "TEXT:\n"
        f"{raw_text}\n"
    )

    inputs = processor(text=repair_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=240,
            do_sample=False,
            num_beams=1
        )

    repaired = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    json_str = _extract_first_json_obj(repaired)
    if not json_str:
        raise ValueError("repair_failed_no_json_object_found")
    return _safe_json_loads(json_str)


def llava_describe_structured(image: Image.Image, max_new_tokens: int = 260) -> Dict[str, Any]:
    """
    Generate artwork description using LLaVA, returning a structured dict.
    On parse errors, returns dict with raw_output + parse_error.
    """
    prompt = "<image>\n" + INSTRUCTION
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )

    raw = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    # Extract JSON object from raw
    json_str = _extract_first_json_obj(raw)
    if not json_str:
        return {
            "raw_output": raw,
            "parse_error": "no_json_object_found"
        }

    try:
        parsed = _safe_json_loads(json_str)
        return parsed
    except Exception as e:
        # Optional repair attempt
        if ENABLE_REPAIR_ATTEMPT:
            try:
                repaired = _repair_to_json(raw_text=raw)
                repaired["_repaired"] = True
                return repaired
            except Exception as e2:
                return {
                    "raw_output": raw,
                    "parse_error": f"json_parse_failed: {type(e).__name__}: {e}",
                    "repair_error": f"{type(e2).__name__}: {e2}"
                }

        return {
            "raw_output": raw,
            "parse_error": f"json_parse_failed: {type(e).__name__}: {e}"
        }


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append records to a JSONL file safely."""
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_to_json_array(jsonl_path: str, json_path: str) -> None:
    """Convert JSONL to a standard JSON array file (for JSONCrack)."""
    data: List[Dict[str, Any]] = []
    if not os.path.exists(jsonl_path):
        return
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# Find images
# =========================
print("\n[2/4] Scanning for images...")
img_exts = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")

if not os.path.isdir(IMAGES_DIR):
    print(f"ERROR: Images directory not found: {IMAGES_DIR}")
    print("Please upload your images to /workspace/images/")
    raise SystemExit(1)

# Load checkpoint: only those with status == ok or ok_with_parse_issue are considered done.
done = set()
if os.path.exists(OUTPUT_FILE_JSONL):
    try:
        with open(OUTPUT_FILE_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                status = str(record.get("status", ""))
                if status in ("ok", "ok_with_parse_issue"):
                    done.add(record.get("image_filename"))
        print(f"✓ Checkpoint loaded: {len(done)} images already processed")
    except Exception as e:
        print(f"⚠ Could not load checkpoint: {e}")
        done = set()

files = sorted([fn for fn in os.listdir(IMAGES_DIR) if fn.lower().endswith(img_exts)])
todo = [fn for fn in files if fn not in done]

print(f"\n{'='*70}")
print("PROCESSING SUMMARY")
print(f"{'='*70}")
print(f"Total images found : {len(files)}")
print(f"Already processed  : {len(done)}")
print(f"To process now     : {len(todo)}")
print(f"Input directory    : {IMAGES_DIR}")
print(f"Output JSONL       : {OUTPUT_FILE_JSONL}")
print(f"Output JSON        : {OUTPUT_FILE_JSON}")
print(f"Max side           : {MAX_SIDE}")
print(f"Max new tokens     : {MAX_NEW_TOKENS}")
print(f"Flush every        : {FLUSH_EVERY}")
print(f"Repair attempt     : {ENABLE_REPAIR_ATTEMPT}")
print(f"{'='*70}\n")

if len(todo) == 0:
    print("✓ All images already processed!")
    # Still ensure JSON array exists for convenience
    jsonl_to_json_array(OUTPUT_FILE_JSONL, OUTPUT_FILE_JSON)
    raise SystemExit(0)

# =========================
# Process images
# =========================
print("[3/4] Processing images...")
rows_buffer: List[Dict[str, Any]] = []
start_time = time.time()

for idx, fn in enumerate(tqdm(todo, desc="Processing"), 1):
    path = os.path.join(IMAGES_DIR, fn)
    t0 = time.time()

    try:
        img = Image.open(path).convert("RGB")
        img = resize_for_vlm(img, max_side=MAX_SIDE)

        desc_obj = llava_describe_structured(img, max_new_tokens=MAX_NEW_TOKENS)
        elapsed = round(time.time() - t0, 2)

        status = "ok"
        if "parse_error" in desc_obj:
            status = "ok_with_parse_issue"

        record = {
            "image_filename": fn,
            "llava_description": desc_obj,  # <--- dict, not string
            "processing_time_seconds": elapsed,
            "status": status,
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
        print(f"\n✗ Error processing {fn}: {type(e).__name__}: {str(e)[:160]}")

        rows_buffer.append({
            "image_filename": fn,
            "llava_description": {"raw_output": "", "parse_error": f"runtime_error: {type(e).__name__}: {e}"},
            "processing_time_seconds": elapsed,
            "status": "error",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embeddings": None,
            "web_metadata": None,
            "similarity_scores": None
        })

    # Flush buffer
    if len(rows_buffer) >= FLUSH_EVERY:
        append_jsonl(OUTPUT_FILE_JSONL, rows_buffer)
        rows_buffer = []

# Final flush
if rows_buffer:
    append_jsonl(OUTPUT_FILE_JSONL, rows_buffer)

# Build JSON array for JSONCrack
jsonl_to_json_array(OUTPUT_FILE_JSONL, OUTPUT_FILE_JSON)

total_time = time.time() - start_time
print(f"\n{'='*70}")
print("[4/4] PROCESSING COMPLETE!")
print(f"{'='*70}")
print(f"Total time        : {total_time/3600:.2f} hours")
print(f"Images processed  : {len(todo)}")
print(f"Output JSONL      : {OUTPUT_FILE_JSONL}")
print(f"Output JSON       : {OUTPUT_FILE_JSON}")
print(f"{'='*70}\n")
print("Download your results from /workspace/output/")