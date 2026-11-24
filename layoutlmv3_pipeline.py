
"""
Insurance PDF extraction pipeline.

Features:
- Native PDF text extraction
- Tesseract OCR fallback
- Word coordinates
- Line and block information
- Spatial row reconstruction
- Native PDF table detection
- OCR confidence
- Pretrained LayoutLMv3 feature inference

LayoutLMv3-base is a feature encoder.
It does not predict trained insurance field labels.
"""

from pathlib import Path
import json
import re
import shutil

import fitz
from PIL import Image
import pytesseract



# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_ID = "microsoft/layoutlmv3-base"

RENDER_SCALE = 2.0

OCR_LANGUAGE = "eng"

OCR_CONFIG = "--oem 3 --psm 3"


# ==========================================
# 2. TESSERACT CONFIGURATION
# ==========================================

def configure_tesseract():

    installed = shutil.which("tesseract")

    if installed:

        pytesseract.pytesseract.tesseract_cmd = installed

        return True

    possible_paths = [
        Path(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        ),

        Path(
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        )
    ]

    for path in possible_paths:

        if path.is_file():

            pytesseract.pytesseract.tesseract_cmd = str(path)

            return True

    return False


# ==========================================
# 3. LOAD LAYOUTLMV3 MODEL
# ==========================================

def load_model(model_dir=None, device=None):
    import torch
    from transformers import LayoutLMv3Processor, LayoutLMv3Model

    if model_dir and Path(model_dir).is_dir():

        source = str(model_dir)

    else:

        source = MODEL_ID

    processor = LayoutLMv3Processor.from_pretrained(
        source,
        apply_ocr=False
    )

    model = LayoutLMv3Model.from_pretrained(
        source
    )

    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    model.eval()

    return processor, model, device


# ==========================================
# 4. NORMALIZE BOUNDING BOX
# ==========================================

def norm_box(box, width, height):

    if width <= 0 or height <= 0:

        return [0, 0, 0, 0]

    x0, y0, x1, y1 = box

    values = [
        x0 / width * 1000,
        y0 / height * 1000,
        x1 / width * 1000,
        y1 / height * 1000
    ]

    return [
        max(0, min(1000, int(round(value))))
        for value in values
    ]


def valid_box(box):

    if len(box) != 4:

        return False

    x0, y0, x1, y1 = box

    return (
        0 <= x0 < x1 <= 1000
        and 0 <= y0 < y1 <= 1000
    )


def page_box(page, box):

    rect = fitz.Rect(box)

    # Native PDF coordinates are unrotated.
    # Transform to the rendered page orientation.
    if page.rotation:

        rect = rect * page.rotation_matrix

    return norm_box(
        (
            rect.x0 - page.rect.x0,
            rect.y0 - page.rect.y0,
            rect.x1 - page.rect.x0,
            rect.y1 - page.rect.y0
        ),
        page.rect.width,
        page.rect.height
    )


def union_boxes(boxes):

    if not boxes:

        return None

    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes)
    ]


# ==========================================
# 5. RENDER PDF PAGE
# ==========================================

def render_pdf_page(page, scale=RENDER_SCALE):

    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csRGB,
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples
    )

    return image


# ==========================================
# 6. CHECK NATIVE TEXT QUALITY
# ==========================================

def native_text_is_valid(text, words):

    compact = re.sub(
        r"\s+",
        "",
        text or ""
    )

    if len(compact) < 30:

        return False

    if len(words) < 5:

        return False

    readable = sum(
        char.isalnum()
        for char in compact
    )

    ratio = readable / max(
        len(compact),
        1
    )

    if ratio < 0.40:

        return False

    replacement_ratio = (
        compact.count("\ufffd")
        / max(len(compact), 1)
    )

    if replacement_ratio > 0.05:

        return False

    return True


# ==========================================
# 7. NATIVE PDF WORD EXTRACTION
# ==========================================

def extract_native_words(page):

    native = page.get_text(
        "words",
        sort=True
    )

    words = []

    for item in native:

        (
            x0,
            y0,
            x1,
            y1,
            token,
            block,
            line,
            word_number
        ) = item

        token = str(token).strip()

        if not token:

            continue

        bbox = page_box(
            page,
            (x0, y0, x1, y1)
        )

        if not valid_box(bbox):

            continue

        words.append({
            "text": token,
            "bbox": bbox,
            "block": int(block),
            "paragraph": 0,
            "line": int(line),
            "word_number": int(word_number),
            "confidence": None,
            "source": "native_pdf_text"
        })

    return words


# ==========================================
# 8. OCR WORD EXTRACTION
# ==========================================

def extract_ocr_words(image):

    if not configure_tesseract():

        raise RuntimeError(
            "Tesseract OCR is not installed "
            "or its executable was not found."
        )

    data = pytesseract.image_to_data(
        image,
        lang=OCR_LANGUAGE,
        config=OCR_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    words = []

    for index, token in enumerate(data["text"]):

        token = str(token).strip()

        if not token:

            continue

        try:

            confidence = float(
                data["conf"][index]
            )

        except (TypeError, ValueError):

            confidence = -1.0

        if confidence < 0:

            continue

        x = int(data["left"][index])

        y = int(data["top"][index])

        width = int(data["width"][index])

        height = int(data["height"][index])

        bbox = norm_box(
            (
                x,
                y,
                x + width,
                y + height
            ),
            image.width,
            image.height
        )

        if not valid_box(bbox):

            continue

        words.append({
            "text": token,
            "bbox": bbox,
            "block": int(
                data["block_num"][index]
            ),
            "paragraph": int(
                data["par_num"][index]
            ),
            "line": int(
                data["line_num"][index]
            ),
            "word_number": int(
                data["word_num"][index]
            ),
            "confidence": round(confidence, 2),
            "source": "tesseract_ocr"
        })

    return words


# ==========================================
# 9. RECONSTRUCT ORIGINAL LINES
# ==========================================

def build_lines(words):

    groups = {}

    for word in words:

        key = (
            word.get("block", 0),
            word.get("paragraph", 0),
            word.get("line", 0)
        )

        groups.setdefault(
            key,
            []
        ).append(word)

    lines = []

    for key, items in groups.items():

        items = sorted(
            items,
            key=lambda word: word["bbox"][0]
        )

        bbox = union_boxes([
            word["bbox"]
            for word in items
        ])

        confidences = [
            word["confidence"]
            for word in items
            if word.get("confidence") is not None
        ]

        lines.append({
            "block": key[0],
            "paragraph": key[1],
            "line": key[2],
            "text": " ".join(
                word["text"]
                for word in items
            ),
            "bbox": bbox,
            "word_count": len(items),
            "confidence": (
                round(
                    sum(confidences)
                    / len(confidences),
                    2
                )
                if confidences
                else None
            )
        })

    return sorted(
        lines,
        key=lambda line: (
            line["bbox"][1],
            line["bbox"][0]
        )
    )


# ==========================================
# 10. RECONSTRUCT SPATIAL ROWS
# ==========================================

def build_spatial_rows(words):

    groups = []

    ordered = sorted(
        words,
        key=lambda word: (
            (
                word["bbox"][1]
                + word["bbox"][3]
            ) / 2,
            word["bbox"][0]
        )
    )

    for word in ordered:

        box = word["bbox"]

        center = (
            box[1] + box[3]
        ) / 2

        height = box[3] - box[1]

        selected = None

        for group in reversed(groups):

            tolerance = max(
                3,
                min(
                    7,
                    height * 0.4
                )
            )

            if abs(
                group["center"] - center
            ) <= tolerance:

                selected = group

                break

            if center - group["center"] > 15:

                break

        if selected is None:

            groups.append({
                "center": center,
                "words": [word]
            })

        else:

            selected["words"].append(word)

    rows = []

    for row_number, group in enumerate(
        groups,
        start=1
    ):

        items = sorted(
            group["words"],
            key=lambda word: word["bbox"][0]
        )

        bbox = union_boxes([
            word["bbox"]
            for word in items
        ])

        for word in items:

            word["row_id"] = row_number

        rows.append({
            "row_id": row_number,
            "text": " ".join(
                word["text"]
                for word in items
            ),
            "bbox": bbox,
            "y": group["center"],
            "words": items
        })

    return rows


# ==========================================
# 11. NATIVE PDF TABLE EXTRACTION
# ==========================================

def extract_native_tables(page):

    tables = []

    # Older PyMuPDF versions may not
    # support find_tables().
    if not hasattr(page, "find_tables"):

        return tables

    try:

        finder = page.find_tables()

    except Exception:

        return tables

    for table_index, table in enumerate(
        finder.tables,
        start=1
    ):

        try:

            values = table.extract()

            rows = []

            for row_index, row in enumerate(
                table.rows
            ):

                cells = []

                for column_index, cell in enumerate(
                    row.cells
                ):

                    value = ""

                    if row_index < len(values):

                        if column_index < len(
                            values[row_index]
                        ):

                            value = (
                                values[row_index][column_index]
                                or ""
                            )

                    bbox = (
                        page_box(page, cell)
                        if cell is not None
                        else None
                    )

                    cells.append({
                        "row": row_index,
                        "column": column_index,
                        "text": str(value).strip(),
                        "bbox": bbox
                    })

                rows.append({
                    "row": row_index,
                    "cells": cells
                })

            tables.append({
                "table_id": table_index,
                "bbox": page_box(
                    page,
                    table.bbox
                ),
                "row_count": table.row_count,
                "column_count": table.col_count,
                "rows": rows
            })

        except Exception:

            continue

    return tables


# ==========================================
# 12. COMPLETE PAGE EXTRACTION
# ==========================================

def extract_page(page, scale=RENDER_SCALE):

    image = render_pdf_page(
        page,
        scale
    )

    native_text = page.get_text(
        "text",
        sort=True
    )

    native_words = extract_native_words(
        page
    )

    if native_text_is_valid(
        native_text,
        native_words
    ):

        words = native_words

        text = native_text

        method = "native_pdf_text"

    else:

        words = extract_ocr_words(
            image
        )

        original_lines = build_lines(
            words
        )

        text = "\n".join(
            line["text"]
            for line in original_lines
        )

        method = "tesseract_ocr"

    # Preserve the original function contract.
    return image, words, text, method


# ==========================================
# 13. TOKEN-AWARE CHUNK SPLITTING
# ==========================================

def split_word_chunks(
    words,
    processor,
    chunk_size=100,
    max_tokens=512
):

    tokenizer = processor.tokenizer

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        while end > start:

            chunk = words[start:end]

            encoding = tokenizer(
                [word["text"] for word in chunk],
                boxes=[
                    word["bbox"]
                    for word in chunk
                ],
                truncation=False,
                add_special_tokens=True
            )

            token_count = len(
                encoding["input_ids"]
            )

            if token_count <= max_tokens:

                break

            if end - start == 1:

                break

            end = start + max(
                1,
                (end - start) // 2
            )

        yield start, end, words[start:end]

        start = end


# ==========================================
# 14. LAYOUTLMV3 FEATURE INFERENCE
# ==========================================

def infer_layoutlmv3(
    image,
    words,
    processor,
    model,
    device,
    chunk_size=100
):

    summaries = []
    if processor is None or model is None:
        return summaries
    import torch

    if not words:

        return summaries

    for start, end, chunk in split_word_chunks(
        words,
        processor,
        chunk_size=chunk_size
    ):

        inputs = processor(
            image,
            [word["text"] for word in chunk],
            boxes=[
                word["bbox"]
                for word in chunk
            ],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        token_count = int(
            inputs["attention_mask"][0]
            .sum()
            .item()
        )

        word_ids = inputs.word_ids(
            batch_index=0
        )

        represented_words = len({
            word_id
            for word_id in word_ids
            if word_id is not None
        })

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            result = model(**inputs)

        summaries.append({
            "word_start": start,
            "word_end_exclusive": end,
            "input_words": len(chunk),
            "represented_words": represented_words,
            "nonpadding_tokens": token_count,
            "hidden_state_shape": list(
                result.last_hidden_state.shape
            ),
            "device": str(device),
            "truncated": (
                represented_words < len(chunk)
            ),
            "note": (
                "Pretrained encoder features, "
                "not semantic insurance field labels"
            )
        })

        del inputs

        del result

    return summaries


# ==========================================
# 15. OCR QUALITY SUMMARY
# ==========================================

def extraction_quality(words, method):

    confidences = [
        word["confidence"]
        for word in words
        if word.get("confidence") is not None
    ]

    if method == "native_pdf_text":

        return {
            "method": method,
            "average_ocr_confidence": None,
            "low_confidence_words": None
        }

    return {
        "method": method,

        "average_ocr_confidence": (
            round(
                sum(confidences)
                / len(confidences),
                2
            )
            if confidences
            else None
        ),

        "low_confidence_words": sum(
            confidence < 60
            for confidence in confidences
        )
    }


# ==========================================
# 16. SAVE JSON
# ==========================================

def save_json(path, data):

    Path(path).write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ==========================================
# 17. PROCESS COMPLETE PDF
# ==========================================

def process_pdf(
    pdf_path,
    output_root,
    processor,
    model,
    device,
    max_pages=None
):

    pdf_path = Path(pdf_path)

    output_dir = (
        Path(output_root)
        / pdf_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pages_out = []

    with fitz.open(str(pdf_path)) as doc:

        if doc.needs_pass:

            raise RuntimeError(
                "This PDF is password protected."
            )

        total = len(doc)

        if total == 0:

            raise RuntimeError(
                "The PDF contains no pages."
            )

        limit = (
            min(total, int(max_pages))
            if max_pages is not None
            else total
        )

        if limit <= 0:

            raise ValueError(
                "max_pages must be greater than zero."
            )

        for page_index in range(limit):

            page_number = page_index + 1

            page = doc[page_index]

            # Step 1: Extract PDF words and text.
            image, words, text, method = extract_page(
                page
            )

            # Step 2: Preserve original lines.
            page_lines = build_lines(
                words
            )

            # Step 3: Reconstruct spatial rows.
            spatial_rows = build_spatial_rows(
                words
            )

            # Step 4: Detect native PDF tables.
            tables = (
                extract_native_tables(page)
                if method == "native_pdf_text"
                else []
            )

            # Step 5: Run pretrained LayoutLMv3.
            embeddings_meta = infer_layoutlmv3(
                image,
                words,
                processor,
                model,
                device
            )

            # Step 6: Prepare page metadata.
            page_data = {
                "page": page_number,

                "text": text,

                "method": method,

                "width": float(
                    page.rect.width
                ),

                "height": float(
                    page.rect.height
                ),

                "rotation": int(
                    page.rotation
                ),

                "word_count": len(words),

                "line_count": len(page_lines),

                "row_count": len(spatial_rows),

                "table_count": len(tables),

                "words": words,

                "lines": page_lines,

                "rows": spatial_rows,

                "tables": tables,

                "quality": extraction_quality(
                    words,
                    method
                ),

                "layoutlmv3_inference":
                    embeddings_meta
            }

            pages_out.append(page_data)

            # Step 7: Save page outputs.
            image.save(
                output_dir
                / f"page_{page_number:03d}.png"
            )

            (
                output_dir
                / f"page_{page_number:03d}.txt"
            ).write_text(
                text,
                encoding="utf-8"
            )

            save_json(
                output_dir
                / f"page_{page_number:03d}.json",
                page_data
            )

    # Step 8: Combine complete extracted text.
    full_text = "\n\n".join(
        (
            f'===== PAGE {page["page"]} =====\n'
            f'{page["text"]}'
        )
        for page in pages_out
    )

    (
        output_dir / "full_text.txt"
    ).write_text(
        full_text,
        encoding="utf-8"
    )

    # Step 9: Prepare complete payload.
    payload = {
        "filename": pdf_path.name,

        "page_count": total,

        "processed_pages": len(pages_out),

        "model": MODEL_ID if model is not None else "Not run (rule-based extraction)",

        "device": str(device),

        "total_words": sum(
            page["word_count"]
            for page in pages_out
        ),

        "total_tables": sum(
            page["table_count"]
            for page in pages_out
        ),

        "native_pages": sum(
            page["method"] == "native_pdf_text"
            for page in pages_out
        ),

        "ocr_pages": sum(
            page["method"] == "tesseract_ocr"
            for page in pages_out
        ),

        "warning": (
            "LayoutLMv3-base produces pretrained "
            "features only. Structured insurance "
            "fields are extracted using separate "
            "rules and validation."
        ),

        "pages": pages_out
    }

    # Step 10: Save final JSON.
    save_json(
        output_dir / "extraction.json",
        payload
    )

    return payload, output_dir
