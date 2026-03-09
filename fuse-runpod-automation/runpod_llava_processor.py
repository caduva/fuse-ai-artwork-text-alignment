#!/usr/bin/env python3
"""
RunPod LLaVA Image Processing Script
Processes artwork images and generates structured descriptions using LLaVA.

Outputs:
- JSONL: /workspace/output/llava_captions.jsonl  (one JSON object per line)
- JSON : /workspace/output/llava_captions.json   (array JSON for JSONCrack)

This version keeps the current single-GPU/stable runtime setup and makes parsing
more resilient for LLaVA-style pseudo-JSON outputs. The main goal is to recover
`objective`, `interpretation`, and `visual_cues` reliably even when the model
does not emit strict JSON.
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

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "350"))
FLUSH_EVERY = int(os.getenv("FLUSH_EVERY", "1"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "768"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))
REPAIR_MAX_NEW_TOKENS = int(os.getenv("REPAIR_MAX_NEW_TOKENS", "350"))

ENABLE_REPAIR_ATTEMPT = int(os.getenv("ENABLE_REPAIR_ATTEMPT", "0"))

# =========================
# Prompt
# =========================
INSTRUCTION = (
    "You are analysing an artwork image.\n"
    "\n"
    "Step 1: Describe objectively what is visibly present in the image.\n"
    "Step 2: Provide a detailed conceptual interpretation of the possible meaning, symbolism, or artistic intent.\n"
    "Base this interpretation on the visual elements that can be observed.\n"
    "Step 3: List 3–5 short visual cues that support your interpretation.\n"
    "\n"
    "IMPORTANT:\n"
    "If there is clearly readable text in the image, treat it as part of the artwork and consider its meaning.\n"
    "Do not invent unreadable text.\n"
    "\n"
    "Do not guess the artist, title, or date.\n"
    "Do not invent facts not supported by the image.\n"
    "\n"
    "Output JSON with keys:\n"
    "objective, interpretation, visual_cues.\n"
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
    Extract the first JSON object substring from model output using balanced braces.
    """
    t = (text or "").strip()
    if not t:
        return None

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
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def _split_short_cues(text: str) -> List[str]:
    """
    Turn a long cue string into a cleaner list of short cues.
    """
    t = (text or "").strip()
    if not t:
        return []

    # Remove common leading labels
    t = re.sub(r"^\s*(visual[_ ]?cues?|cues?)\s*:\s*", "", t, flags=re.I)

    # First pass: bullet-like separators and sentence boundaries
    parts = re.split(r"(?:\s*[;\n•]\s*|\.\s+)", t)
    cleaned = []
    for part in parts:
        part = part.strip(" -,\t\r\n")
        if not part:
            continue
        # Second pass: break long conjunction chains
        subparts = re.split(r"\s+(?:and|with|plus)\s+", part)
        for sp in subparts:
            sp = sp.strip(" -,\t\r\n")
            if sp:
                cleaned.append(sp)

    # Deduplicate while preserving order
    out = []
    seen = set()
    for item in cleaned:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)

    return out[:8]


def _normalise_to_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                items.append(s)
        if field_name == "visual_cues" and len(items) == 1:
            return _split_short_cues(items[0]) or items
        return items

    s = str(value).strip()
    if not s:
        return []
    if field_name == "visual_cues":
        return _split_short_cues(s) or [s]
    return [s]


def _normalise_llava_json(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(d or {})

    out.setdefault("objective", "")
    out.setdefault("interpretation", "")
    out.setdefault("visual_cues", [])

    out["objective"] = str(out.get("objective") or "").strip()
    out["interpretation"] = str(out.get("interpretation") or "").strip()
    out["visual_cues"] = _normalise_to_list(out.get("visual_cues"), "visual_cues")

    # Remove obvious prompt-echo artefacts
    bad_objectives = {
        "analyse an artwork image",
        "analyze an artwork image",
        "analyse the artwork image",
        "analyze the artwork image",
    }
    if out["objective"].strip().lower() in bad_objectives:
        out["objective"] = ""

    if out["interpretation"].strip().lower() in bad_objectives:
        out["interpretation"] = ""

    return out


def _extract_section(raw: str, key: str, next_keys: List[str]) -> str:
    """
    Extract a section from pseudo-JSON or key:value text such as:
    objective: ...
    interpretation: ...
    visual_cues: ...
    """
    if not raw:
        return ""

    # Build a forgiving boundary pattern
    if next_keys:
        next_group = "|".join(re.escape(k) for k in next_keys)
        pattern = rf"(?is)(?:^|\n|\r)\s*{re.escape(key)}\s*[:=]\s*(.*?)(?=(?:\n|\r)\s*(?:{next_group})\s*[:=]|\Z)"
        m = re.search(pattern, raw)
        if not m:
            pattern_inline = rf"(?is)\b{re.escape(key)}\s*[:=]\s*(.*?)(?=\b(?:{next_group})\s*[:=]|\Z)"
            m = re.search(pattern_inline, raw)
            if not m:
                return ""
        return m.group(1).strip(" \n\r\t,;")
    else:
        pattern = rf"(?is)(?:^|\n|\r)\s*{re.escape(key)}\s*[:=]\s*(.*?)(?=\Z)"
        m = re.search(pattern, raw)
        if not m:
            pattern_inline = rf"(?is)\b{re.escape(key)}\s*[:=]\s*(.*?)(?=\Z)"
            m = re.search(pattern_inline, raw)
            if not m:
                return ""
        return m.group(1).strip(" \n\r\t,;")


def _parse_loose_key_value_output(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse non-JSON outputs that still contain the desired keys.
    Works on formats such as:
    objective: ...
    interpretation: ...
    visual_cues: ...
    """
    if not raw:
        return None

    text = raw.replace("\\_", "_").strip()

    # Strip fenced code blocks if any
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    extracted = {
        "objective": _extract_section(text, "objective", ["interpretation", "visual_cues"]),
        "interpretation": _extract_section(text, "interpretation", ["visual_cues"]),
        "visual_cues": _extract_section(text, "visual_cues", []),
    }

    # If not enough structure, give up
    if not any(extracted.values()):
        return None

    # Handle possible JSON-like list strings
    val = extracted["visual_cues"]
    if val:
        v = str(val).strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                extracted["visual_cues"] = json.loads(v)
            except Exception:
                pass

    return _normalise_llava_json(extracted)


def _cleanup_cuda(*objs: Any) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _move_inputs_to_device(batch: Dict[str, Any], target_device: str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(target_device)
        else:
            moved[k] = v
    return moved


def _repair_to_json(raw_text: str) -> Dict[str, Any]:
    repair_prompt = (
        "You will be given an output that SHOULD be a single JSON object.\n"
        "Fix it and output STRICT JSON only.\n\n"
        "Rules:\n"
        "- Use only these keys: objective, interpretation, visual_cues.\n"
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
    Generate artwork description using LLaVA.

    Success hierarchy:
    1. strict JSON parse
    2. loose key-value parse
    3. optional repair pass
    4. raw fallback

    Success is defined by usable recovery of objective + interpretation +
    visual_cues, with interpretation being the most important field.
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

    # Path 1: strict JSON object
    json_str = _extract_first_json_obj(raw)
    if json_str:
        try:
            parsed = _safe_json_loads(json_str)
            parsed = _normalise_llava_json(parsed)
            parsed["_parse_mode"] = "strict_json"
            return parsed
        except Exception as e:
            strict_error = f"json_parse_failed: {type(e).__name__}: {e}"
        else:
            strict_error = None
    else:
        strict_error = "no_json_object_found"

    # Path 2: loose key:value extraction
    loose = _parse_loose_key_value_output(raw)
    if loose and loose.get("objective") and loose.get("interpretation") and loose.get("visual_cues"):
        loose["_parse_mode"] = "loose_key_value"
        if strict_error:
            loose["_parse_note"] = strict_error
        return loose

    # Path 3: optional repair pass
    if ENABLE_REPAIR_ATTEMPT:
        try:
            repaired = _repair_to_json(raw_text=raw)
            repaired = _normalise_llava_json(repaired)
            repaired["_repaired"] = True
            repaired["_parse_mode"] = "repair_pass"
            if strict_error:
                repaired["_parse_note"] = strict_error
            return repaired
        except Exception as e2:
            return {
                "raw_output": raw,
                "parse_error": strict_error or "unknown_parse_error",
                "repair_error": f"{type(e2).__name__}: {e2}",
            }

    # Path 4: raw fallback
    return {
        "raw_output": raw,
        "parse_error": strict_error or "unknown_parse_error",
    }


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def jsonl_to_json_array(jsonl_path: str, json_path: str) -> None:
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


print("\n[1/4] Preparing output directory...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

print("\n[2/4] Scanning for images...")
img_exts = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")

if not os.path.isdir(IMAGES_DIR):
    print(f"ERROR: Images directory not found: {IMAGES_DIR}")
    print("Please upload your images to /workspace/images/")
    raise SystemExit(1)

# Load checkpoint: treat records with usable structured output as done too.
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
                desc = record.get("llava_description") or {}
                if status in ("ok", "ok_with_parse_issue", "ok_loose_parse"):
                    done.add(record.get("image_filename"))
                elif isinstance(desc, dict):
                    has_objective = bool(str(desc.get("objective", "")).strip())
                    has_interpretation = bool(str(desc.get("interpretation", "")).strip())
                    has_visual_cues = bool(desc.get("visual_cues", []))
                    if has_objective and has_interpretation and has_visual_cues:
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

        parse_mode = llava_json.get("_parse_mode") if isinstance(llava_json, dict) else None
        has_objective = isinstance(llava_json, dict) and bool(str(llava_json.get("objective", "")).strip())
        has_interpretation = isinstance(llava_json, dict) and bool(str(llava_json.get("interpretation", "")).strip())
        visual_cues = llava_json.get("visual_cues", []) if isinstance(llava_json, dict) else []
        has_visual_cues = bool(visual_cues)

        if has_objective and has_interpretation and has_visual_cues and parse_mode == "loose_key_value":
            status = "ok_loose_parse"
        elif has_objective and has_interpretation and has_visual_cues:
            status = "ok"
        elif "parse_error" in llava_json:
            status = "ok_with_parse_issue"
        else:
            status = "ok"

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

if buffer:
    append_jsonl(OUTPUT_FILE_JSONL, buffer)

print("\n[4/4] Building JSON array file...")
jsonl_to_json_array(OUTPUT_FILE_JSONL, OUTPUT_FILE_JSON)

print("\nDone.")
