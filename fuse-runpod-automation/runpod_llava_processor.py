#!/usr/bin/env python3
"""
RunPod LLaVA Image Processing Script
Processes artwork images and generates structured descriptions using LLaVA.

Outputs:
- JSONL: /workspace/output/llava_captions.jsonl  (one JSON object per line)
- JSON : /workspace/output/llava_captions.json   (array JSON for JSONCrack)

Final production-oriented version:
- Keeps the same model, prompt, image size, and token budget used for quality.
- Flushes every image to reduce loss on interruptions.
- Loads the model explicitly onto a single GPU when CUDA is available.
- Uses torch.inference_mode() and explicit gc cleanup for long runs.
"""

import gc
import os
import json
import time
import re
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
FLUSH_EVERY = int(os.getenv("FLUSH_EVERY", "1"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "768"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))
REPAIR_MAX_NEW_TOKENS = int(os.getenv("REPAIR_MAX_NEW_TOKENS", "240"))

ENABLE_REPAIR_ATTEMPT = int(os.getenv("ENABLE_REPAIR_ATTEMPT", "0"))

# =========================
# Prompt
# =========================
INSTRUCTION = (
    "You are analysing an artwork image.\n"
    "\n"
    "Step 1: Describe objectively what is visibly present in the image.\n"
    "Step 2: Provide a conceptual interpretation of the possible meaning, symbolism, or artistic intent.\n"
    "Base this interpretation on the visual elements that can be observed.\n"
    "Step 3: List 4–8 short visual cues that support your interpretation.\n"
    "Step 4: Offer up to 2 alternative interpretations if plausible.\n"
    "\n"
    "IMPORTANT:\n"
    "If there is clearly readable text in the image, treat it as part of the artwork and consider its meaning.\n"
    "Do not invent unreadable text.\n"
    "\n"
    "Do not guess the artist, title, or date.\n"
    "Do not invent facts not supported by the image.\n"
    "\n"
    "Output JSON with keys:\n"
    "objective, interpretation, visual_cues, alternatives, confidence.\n"
)

# Globals initialised after model load
processor = None
model = None
device = "cpu"


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

    Uses a balanced-brace scan to avoid greedy regex issues when the model
    outputs extra braces or multiple JSON objects.
    """
    t = (text or "").strip()
    if not t:
        return None

    # Normalise common artefact: model outputs 'visual\_cues' (escaped underscore)
    t = t.replace("\\_", "_")

    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]

    return None


def _safe_json_loads(s: str) -> Dict[str, Any]:
    """Load JSON safely, raising a useful error message."""
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def _normalise_llava_json(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise model output to a stable schema for downstream use.
    - Ensures required keys exist.
    - Coerces visual_cues/alternatives to lists.
    - Coerces confidence to float in [0, 1] when possible.
    """
    out: Dict[str, Any] = dict(d or {})

    out.setdefault("objective", "")
    out.setdefault("interpretation", "")
    out.setdefault("visual_cues", [])
    out.setdefault("alternatives", [])
    out.setdefault("confidence", None)

    if isinstance(out.get("visual_cues"), str):
        vc = out["visual_cues"].strip()
        out["visual_cues"] = [vc] if vc else []
    elif out.get("visual_cues") is None:
        out["visual_cues"] = []
    else:
        out["visual_cues"] = list(out["visual_cues"]) if not isinstance(out["visual_cues"], list) else out["visual_cues"]

    if isinstance(out.get("alternatives"), str):
        alt = out["alternatives"].strip()
        out["alternatives"] = [alt] if alt else []
    elif out.get("alternatives") is None:
        out["alternatives"] = []
    else:
        out["alternatives"] = list(out["alternatives"]) if not isinstance(out["alternatives"], list) else out["alternatives"]

    conf = out.get("confidence")
    if isinstance(conf, str):
        s = conf.strip().lower()
        mapping = {
            "very high": 0.95,
            "high": 0.85,
            "medium": 0.65,
            "moderate": 0.65,
            "low": 0.35,
            "very low": 0.15,
        }
        if s in mapping:
            out["confidence"] = mapping[s]
        else:
            try:
                out["confidence"] = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)[0])
            except Exception:
                out["confidence"] = None
    elif isinstance(conf, (int, float)):
        out["confidence"] = float(conf)
    else:
        out["confidence"] = None

    if isinstance(out["confidence"], float):
        out["confidence"] = max(0.0, min(1.0, out["confidence"]))

    if isinstance(out.get("objective"), str):
        obj = out["objective"].strip()
        if obj.lower() in (
            "analyse an artwork image",
            "analyze an artwork image",
            "analyse the artwork image",
            "analyze the artwork image",
        ):
            out["objective"] = ""

    return out


def _cleanup_cuda(*objs: Any) -> None:
    """Free references and encourage host/GPU memory cleanup."""
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _move_inputs_to_device(batch: Dict[str, Any], target_device: str) -> Dict[str, Any]:
    """Move tensor inputs to the selected device."""
    moved: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(target_device)
        else:
            moved[k] = v
    return moved


def _repair_to_json(raw_text: str) -> Dict[str, Any]:
    """Attempt to repair malformed model output into strict JSON using a second pass."""
    repair_prompt = (
        "You will be given an output that SHOULD be a single JSON object.\n"
        "Fix it and output STRICT JSON only.\n\n"
        "Rules:\n"
        "- Do NOT include any extra keys.\n"
        "- Do NOT include any markdown.\n\n"
        "TEXT:\n"
        f"{raw_text}\n"
    )

    inputs = processor(text=repair_prompt, return_tensors="pt")
    inputs = _move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=REPAIR_MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )

    repaired = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    _cleanup_cuda(out, inputs)

    json_str = _extract_first_json_obj(repaired)
    if not json_str:
        raise ValueError("repair_failed_no_json_object_found")
    return _normalise_llava_json(_safe_json_loads(json_str))


def llava_describe_structured(image: Image.Image, max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, Any]:
    """
    Generate artwork description using LLaVA, returning a structured dict.
    On parse errors, returns dict with raw_output + parse_error.
    """
    prompt = "<image>\n" + INSTRUCTION
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = _move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=NUM_BEAMS,
        )

    raw = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    _cleanup_cuda(out, inputs)

    json_str = _extract_first_json_obj(raw)
    if not json_str:
        return {
            "raw_output": raw,
            "parse_error": "no_json_object_found",
        }

    try:
        parsed = _safe_json_loads(json_str)
        return _normalise_llava_json(parsed)
    except Exception as e:
        if ENABLE_REPAIR_ATTEMPT:
            try:
                repaired = _repair_to_json(raw_text=raw)
                repaired = _normalise_llava_json(repaired)
                repaired["_repaired"] = True
                return repaired
            except Exception as e2:
                return {
                    "raw_output": raw,
                    "parse_error": f"json_parse_failed: {type(e).__name__}: {e}",
                    "repair_error": f"{type(e2).__name__}: {e2}",
                }

        return {
            "raw_output": raw,
            "parse_error": f"json_parse_failed: {type(e).__name__}: {e}",
        }


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append records to a JSONL file safely."""
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


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
# [1/4] Setup output dir
# =========================
print("\n[1/4] Preparing output directory...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load model + processor
# =========================
print("\n[1/4] Loading model + processor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_ID)

_model_kwargs = dict(
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
)

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    **_model_kwargs,
)

if device == "cuda":
    model = model.to("cuda")

model.eval()

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
print(f"Beams              : {NUM_BEAMS}")
print(f"Flush every        : {FLUSH_EVERY}")
print(f"Repair attempt     : {ENABLE_REPAIR_ATTEMPT}")
print(f"{'='*70}\n")

if len(todo) == 0:
    print("Nothing to do.")
    raise SystemExit(0)

# =========================
# Process
# =========================
print("\n[3/4] Processing images...")

buffer: List[Dict[str, Any]] = []
processed_now = 0

for fn in tqdm(todo):
    in_path = os.path.join(IMAGES_DIR, fn)
    t0 = time.time()
    img = None

    try:
        img = Image.open(in_path).convert("RGB")
        img = resize_for_vlm(img, MAX_SIDE)

        llava_json = llava_describe_structured(img, max_new_tokens=MAX_NEW_TOKENS)

        status = "ok"
        if "parse_error" in llava_json:
            status = "ok_with_parse_issue"

        record = {
            "image_filename": fn,
            "status": status,
            "llava_description": llava_json,
            "elapsed_seconds": round(time.time() - t0, 3),
            "model_id": MODEL_ID,
        }
    except Exception as e:
        record = {
            "image_filename": fn,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "elapsed_seconds": round(time.time() - t0, 3),
            "model_id": MODEL_ID,
        }
    finally:
        _cleanup_cuda(img)

    buffer.append(record)
    processed_now += 1

    if processed_now % FLUSH_EVERY == 0:
        append_jsonl(OUTPUT_FILE_JSONL, buffer)
        buffer = []

# Flush remaining
if buffer:
    append_jsonl(OUTPUT_FILE_JSONL, buffer)

# Convert to JSON array for convenience
print("\n[4/4] Building JSON array file...")
jsonl_to_json_array(OUTPUT_FILE_JSONL, OUTPUT_FILE_JSON)

print("\nDone.")
