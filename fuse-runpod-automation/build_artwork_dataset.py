from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ========= CONFIG =========
INPUT_JSON = Path(r"C:\Users\ctucunduva\fontes fuse\fuse-ai-artwork-text-alignment\fuse-runpod-automation\out\llava_captions.json")
OUTPUT_DIR = Path(r"C:\Users\ctucunduva\fontes fuse\fuse-ai-artwork-text-alignment\fuse-runpod-automation\out")

OUTPUT_XLSX = OUTPUT_DIR / "artwork_descriptions_dataset.xlsx"
OUTPUT_JSON = OUTPUT_DIR / "artwork_descriptions_dataset.json"
OUTPUT_JSONL = OUTPUT_DIR / "artwork_descriptions_dataset.jsonl"

VALID_STATUSES = {"ok", "ok_loose_parse"}
# ==========================


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return " ".join(text.split()).strip()


def build_description(objective: str, interpretation: str) -> str:
    parts = [clean_text(objective), clean_text(interpretation)]
    return " ".join(part for part in parts if part).strip()


def safe_get_description_block(row: dict[str, Any]) -> dict[str, Any]:
    block = row.get("llava_description", {})
    if isinstance(block, dict):
        return block
    return {}


def main() -> None:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_JSON}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_JSON.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("O arquivo JSON de entrada não contém uma lista de registros.")

    dataset_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for row in raw_data:
        image_filename = clean_text(row.get("image_filename", ""))
        status = clean_text(row.get("status", ""))
        elapsed_seconds = row.get("elapsed_seconds")
        model_id = clean_text(row.get("model_id", ""))

        block = safe_get_description_block(row)
        objective = clean_text(block.get("objective", ""))
        interpretation = clean_text(block.get("interpretation", ""))
        visual_cues = block.get("visual_cues", [])
        parse_mode = clean_text(block.get("_parse_mode", ""))
        parse_note = clean_text(block.get("_parse_note", ""))

        if isinstance(visual_cues, list):
            visual_cues_text = " | ".join(clean_text(x) for x in visual_cues if clean_text(x))
        else:
            visual_cues_text = clean_text(visual_cues)

        description = build_description(objective, interpretation)

        is_valid_status = status in VALID_STATUSES
        has_minimum_content = bool(objective and interpretation)

        if is_valid_status and has_minimum_content:
            dataset_rows.append({
                "image": image_filename,
                "objective": objective,
                "interpretation": interpretation,
                "description": description,
                "visual_cues": visual_cues_text,
                "status": status,
                "parse_mode": parse_mode,
                "parse_note": parse_note,
                "elapsed_seconds": elapsed_seconds,
                "model_id": model_id,
            })
        else:
            skipped_rows.append({
                "image": image_filename,
                "status": status,
                "objective": objective,
                "interpretation": interpretation,
                "reason": (
                    "invalid_status"
                    if not is_valid_status
                    else "missing_objective_or_interpretation"
                ),
                "parse_mode": parse_mode,
                "parse_note": parse_note,
                "elapsed_seconds": elapsed_seconds,
                "model_id": model_id,
            })

    df_dataset = pd.DataFrame(dataset_rows)
    df_skipped = pd.DataFrame(skipped_rows)

    # Ordena por imagem para facilitar conferência
    if not df_dataset.empty and "image" in df_dataset.columns:
        df_dataset = df_dataset.sort_values(by="image").reset_index(drop=True)

    if not df_skipped.empty and "image" in df_skipped.columns:
        df_skipped = df_skipped.sort_values(by="image").reset_index(drop=True)

    # Salva JSON
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, ensure_ascii=False, indent=2)

    # Salva JSONL
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for item in dataset_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Salva Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_dataset.to_excel(writer, sheet_name="dataset", index=False)
        df_skipped.to_excel(writer, sheet_name="skipped", index=False)

        # Ajuste básico de largura de colunas
        for sheet_name, df in [("dataset", df_dataset), ("skipped", df_skipped)]:
            ws = writer.book[sheet_name]
            for idx, col in enumerate(df.columns, start=1):
                max_len = max(
                    [len(str(col))] +
                    [len(str(v)) for v in df[col].head(200).fillna("").tolist()]
                )
                ws.column_dimensions[chr(64 + idx) if idx <= 26 else ws.cell(row=1, column=idx).column_letter].width = min(max_len + 2, 60)

    print("Processamento concluído.")
    print(f"Entrada: {INPUT_JSON}")
    print(f"Registros válidos: {len(df_dataset)}")
    print(f"Registros descartados: {len(df_skipped)}")
    print(f"Excel gerado em: {OUTPUT_XLSX}")
    print(f"JSON gerado em: {OUTPUT_JSON}")
    print(f"JSONL gerado em: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()