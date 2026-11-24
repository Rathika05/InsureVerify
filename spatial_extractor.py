
"""
Common spatial extraction engine for insurance PDFs.

Supports:
1. Same-line label and value
2. Right-side table cells
3. Values below labels
4. Column-based layouts
5. Multiline values
6. Section context
7. Exact source bounding boxes

Uses normalized PDF word coordinates (0-1000).
No LLM and no insurance-company-specific values.
"""

import re
from functools import lru_cache
from statistics import median


# ==========================================
# 1. TEXT NORMALIZATION
# ==========================================

def normalize_text(text):

    text = str(text or "").lower()

    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def clean_text(text):

    return re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip(" :;|")


# ==========================================
# 2. BOUNDING BOX UTILITIES
# ==========================================

def valid_box(box):

    if not isinstance(box, (list, tuple)):
        return False

    if len(box) != 4:
        return False

    try:

        x0, y0, x1, y1 = map(float, box)

        return (
            0 <= x0 < x1 <= 1000
            and 0 <= y0 < y1 <= 1000
        )

    except (ValueError, TypeError):

        return False


def merge_boxes(boxes):

    boxes = [
        box for box in boxes
        if valid_box(box)
    ]

    if not boxes:
        return None

    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes)
    ]


def center_x(box):

    return (box[0] + box[2]) / 2


def center_y(box):

    return (box[1] + box[3]) / 2


def word_height(word):

    box = word["bbox"]

    return box[3] - box[1]


def prepare_words(page):

    result = []

    for word in page.get("words", []):

        text = str(
            word.get("text", "")
        ).strip()

        box = word.get("bbox")

        if not text or not valid_box(box):
            continue

        result.append({
            **word,
            "text": text,
            "bbox": list(map(float, box))
        })

    return result


def words_text(words):

    return " ".join(
        word["text"]
        for word in words
    )


def words_box(words):

    return merge_boxes([
        word["bbox"]
        for word in words
    ])


# ==========================================
# 3. RECONSTRUCT VISUAL ROWS
# ==========================================

def build_rows(page, y_tolerance=None):

    if y_tolerance is None and "_cached_spatial_rows" in page:
        return page["_cached_spatial_rows"]
    words = prepare_words(page)

    if not words:
        return []

    typical_height = median([
        word_height(word)
        for word in words
    ])

    tolerance = (
        y_tolerance
        if y_tolerance is not None
        else max(2, min(9, typical_height * 0.45))
    )

    words.sort(
        key=lambda word: (
            center_y(word["bbox"]),
            word["bbox"][0]
        )
    )

    groups = []

    for word in words:

        y = center_y(word["bbox"])

        matching = [
            group for group in groups
            if abs(group["y"] - y) <= tolerance
        ]

        if matching:

            group = min(
                matching,
                key=lambda item: abs(item["y"] - y)
            )

            group["words"].append(word)

            group["y"] = median([
                center_y(w["bbox"])
                for w in group["words"]
            ])

        else:

            groups.append({
                "y": y,
                "words": [word]
            })

    rows = []

    for group in groups:

        row_words = sorted(
            group["words"],
            key=lambda word: word["bbox"][0]
        )

        box = words_box(row_words)

        rows.append({
            "text": words_text(row_words),
            "words": row_words,
            "bbox": box,
            "x0": box[0],
            "y0": box[1],
            "x1": box[2],
            "y1": box[3],
            "y": center_y(box)
        })

    rows.sort(
        key=lambda row: (
            row["y"],
            row["x0"]
        )
    )

    for index, row in enumerate(rows):

        row["row_index"] = index

    if y_tolerance is None:
        page["_cached_spatial_rows"] = rows
    return rows


# ==========================================
# 4. SPLIT ROW INTO SPATIAL CELLS
# ==========================================

def split_cells(row, gap_threshold=None):

    words = row.get("words", [])

    if not words:
        return []

    typical_height = median([
        word_height(word)
        for word in words
    ])

    threshold = (
        gap_threshold
        if gap_threshold is not None
        else max(10, typical_height * 1.5)
    )

    groups = [[words[0]]]

    for word in words[1:]:

        previous = groups[-1][-1]

        gap = (
            word["bbox"][0]
            - previous["bbox"][2]
        )

        if gap > threshold:

            groups.append([word])

        else:

            groups[-1].append(word)

    cells = []

    for index, group in enumerate(groups):

        box = words_box(group)

        cells.append({
            "cell_index": index,
            "text": words_text(group),
            "words": group,
            "bbox": box,
            "x0": box[0],
            "y0": box[1],
            "x1": box[2],
            "y1": box[3]
        })

    return cells


# ==========================================
# 5. FIND LABEL WORD SPANS
# ==========================================

@lru_cache(maxsize=256)
def _alias_targets(aliases):
    return frozenset(normalize_text(a) for a in aliases if normalize_text(a))

@lru_cache(maxsize=8192)
def _row_spans(tokens, targets):
    max_length = max((len(t.split()) for t in targets), default=0)
    first = {t.split()[0] for t in targets}
    matches = []
    for start, token in enumerate(tokens):
        n = normalize_text(token)
        if not n or n.split()[0] not in first: continue
        for end in range(start+1, min(len(tokens),start+max_length+3)+1):
            if normalize_text(' '.join(tokens[start:end])) in targets: matches.append((start,end))
    occupied=set(); result=[]
    for start,end in sorted(matches,key=lambda x:(-(x[1]-x[0]),x[0])):
        if not occupied.intersection(range(start,end)):
            occupied.update(range(start,end)); result.append((start,end))
    return tuple(sorted(result))

def find_label_spans(words, aliases):
    targets=_alias_targets(tuple(aliases))
    return [dict(start=a,end=b,label=words_text(words[a:b]),bbox=words_box(words[a:b]),words=words[a:b]) for a,b in _row_spans(tuple(w['text'] for w in words),targets)]

def contains_known_label(text, aliases):

    normalized = normalize_text(text)

    for alias in aliases:

        target = normalize_text(alias)

        if not target:
            continue

        if re.match(
            r"^" + re.escape(target) + r"(?:\s|$)",
            normalized
        ):

            return True

    return False


# ==========================================
# 6. FIND EXACT VALUE COORDINATES
# ==========================================

def find_value_bbox(
    words,
    value,
    near=None,
    max_words=40
):

    def compact(text):

        return re.sub(
            r"[^a-z0-9]",
            "",
            str(text).lower()
        )

    target = compact(value)

    if not target:
        return None

    matches = []

    for start in range(len(words)):

        joined = ""

        for end in range(
            start,
            min(start + max_words, len(words))
        ):

            joined += compact(
                words[end]["text"]
            )

            if joined == target:

                box = words_box(
                    words[start:end + 1]
                )

                if box:
                    matches.append(box)

                break

            if not target.startswith(joined):
                break

    if not matches:
        # A value can be a delimited part of a PDF token (e.g. town-626123).
        # Highlight the containing token; never match an arbitrary substring.
        for word in words:
            raw = str(word['text'])
            if re.search(r'(?<![A-Za-z0-9])' + re.escape(str(value)) + r'(?![A-Za-z0-9])', raw, re.I):
                matches.append(word['bbox'])
    if not matches:
        return None

    if near is not None:

        return min(
            matches,
            key=lambda box: abs(
                center_y(box) - near
            )
        )

    return matches[0]


# ==========================================
# 7. BUILD EXTRACTION CANDIDATE
# ==========================================

def make_candidate(
    page,
    label,
    words,
    method,
    priority,
    evidence
):

    words = [
        word for word in words
        if re.search(
            r"[\w&/-]",
            word["text"]
        )
    ]

    value = clean_text(
        words_text(words)
    )

    box = words_box(words)

    if not value or not box:
        return None

    return {
        "value": value,
        "page": page.get("page"),
        "bbox": box,
        "label": label,
        "method": method,
        "priority": priority,
        "evidence": str(evidence)[:500],
        "words": words
    }


# ==========================================
# 8. FIND VALUE ON THE SAME ROW
# ==========================================

def same_row_candidates(
    page,
    row,
    label,
    all_aliases
):

    words = row["words"]

    end = label["end"]

    other_labels = find_label_spans(
        words,
        all_aliases
    )

    next_label = min(
        [
            item["start"]
            for item in other_labels
            if item["start"] >= end
        ],
        default=len(words)
    )

    tail = words[end:next_label]

    while tail and not re.search(
        r"\w",
        tail[0]["text"]
    ):

        tail = tail[1:]

    if not tail:
        return []

    if contains_known_label(
        words_text(tail),
        all_aliases
    ):
        return []

    candidates = []

    candidate = make_candidate(
        page=page,
        label=label["label"],
        words=tail,
        method="same_row",
        priority=10,
        evidence=row["text"]
    )

    if candidate:

        candidates.append(candidate)

    # Also inspect the nearest separate cell.
    cells = split_cells({
        "words": tail
    })

    if len(cells) > 1:

        candidate = make_candidate(
            page=page,
            label=label["label"],
            words=cells[0]["words"],
            method="right_cell",
            priority=8,
            evidence=row["text"]
        )

        if candidate:

            candidates.append(candidate)

    return candidates


# ==========================================
# 9. FIND VALUE BELOW THE LABEL
# ==========================================

def below_label_candidates(
    page,
    rows,
    row_index,
    label,
    all_aliases,
    max_distance=45
):

    candidates = []

    current = rows[row_index]

    label_box = label["bbox"]

    # If another label exists to the right,
    # use it as a column boundary.
    spans = find_label_spans(
        current["words"],
        all_aliases
    )

    right_boundary = min(
        [
            item["bbox"][0]
            for item in spans
            if item["start"] >= label["end"]
        ],
        default=1001
    )

    for row in rows[row_index + 1:]:

        distance = row["y0"] - current["y1"]

        if distance > max_distance:
            break

        if distance < -3:
            continue

        aligned_words = [
            word for word in row["words"]
            if (
                word["bbox"][0] >= label_box[0] - 15
                and center_x(word["bbox"]) < right_boundary
            )
        ]

        if not aligned_words:
            continue

        if aligned_words[0]["bbox"][0] > label_box[2] + 30:
            continue

        if contains_known_label(
            words_text(aligned_words),
            all_aliases
        ):
            break

        next_labels = find_label_spans(
            aligned_words,
            all_aliases
        )

        if next_labels:

            aligned_words = aligned_words[
                :next_labels[0]["start"]
            ]

        candidate = make_candidate(
            page=page,
            label=label["label"],
            words=aligned_words,
            method="below_label",
            priority=25 + max(0, distance),
            evidence=(
                current["text"]
                + " | "
                + row["text"]
            )
        )

        if candidate:

            candidates.append(candidate)

            # Prefer the nearest plausible row.
            break

    return candidates


# ==========================================
# 10. MULTILINE VALUE EXTRACTION
# ==========================================

def extend_multiline(
    page,
    candidate,
    rows,
    row_index,
    all_aliases,
    max_lines=4,
    max_gap=22
):

    collected = list(
        candidate.get("words", [])
    )

    if not collected:
        return None

    start_box = words_box(collected)

    previous_bottom = start_box[3]

    start_x = start_box[0]

    evidence = [
        candidate["evidence"]
    ]

    added = 0

    for row in rows[row_index + 1:]:

        if added >= max_lines - 1:
            break

        gap = row["y0"] - previous_bottom

        if gap > max_gap:
            break

        if gap < -3:
            continue

        if contains_known_label(
            row["text"],
            all_aliases
        ):
            break

        if re.search(
            r"^\s*(?:page\s+\d+|"
            r"authorised signatory|"
            r"terms and conditions)\b",
            row["text"],
            re.I
        ):
            break

        aligned = [
            word for word in row["words"]
            if word["bbox"][0] >= start_x - 18
        ]

        if not aligned:
            break

        if abs(aligned[0]["bbox"][0] - start_x) > 35:
            break

        labels = find_label_spans(
            aligned,
            all_aliases
        )

        if labels:
            break

        collected.extend(aligned)

        previous_bottom = words_box(aligned)[3]

        evidence.append(row["text"])

        added += 1

    if not added:
        return None

    return make_candidate(
        page=page,
        label=candidate["label"],
        words=collected,
        method="multiline",
        priority=15,
        evidence=" | ".join(evidence)
    )


# ==========================================
# 11. COMMON FIELD CANDIDATE EXTRACTION
# ==========================================

def extract_candidates(
    page,
    aliases,
    all_aliases=None,
    multiline=False
):

    all_aliases = list(
        all_aliases or aliases
    )

    rows = build_rows(page)

    candidates = []

    for index, row in enumerate(rows):

        labels = [span for span in find_label_spans(row["words"], all_aliases) if normalize_text(span["label"]) in _alias_targets(tuple(aliases))]

        for label in labels:

            # Avoid matching short labels inside prose.
            prefix = words_text(
                row["words"][:label["start"]]
            )

            if (
                len(prefix.split()) > 4
                and not find_label_spans(
                    row["words"][:label["start"]],
                    all_aliases
                )
            ):
                continue

            same_row = same_row_candidates(
                page,
                row,
                label,
                all_aliases
            )

            candidates.extend(same_row)

            if multiline:

                for candidate in same_row:

                    extended = extend_multiline(
                        page,
                        candidate,
                        rows,
                        index,
                        all_aliases
                    )

                    if extended:

                        candidates.append(extended)

            below = below_label_candidates(
                page,
                rows,
                index,
                label,
                all_aliases
            )

            candidates.extend(below)

    # Remove duplicate candidates.
    unique = {}

    for candidate in candidates:

        key = (
            normalize_text(candidate["value"]),
            candidate["page"],
            tuple(candidate["bbox"])
        )

        old = unique.get(key)

        if (
            old is None
            or candidate["priority"] < old["priority"]
        ):

            unique[key] = candidate

    return sorted(
        unique.values(),
        key=lambda candidate: candidate["priority"]
    )


# ==========================================
# 12. SECTION IDENTIFICATION
# ==========================================

def find_sections(
    page,
    section_aliases
):

    rows = build_rows(page)

    sections = []

    for index, row in enumerate(rows):

        for section, aliases in section_aliases.items():

            if contains_known_label(
                row["text"],
                aliases
            ):

                sections.append({
                    "section": section,
                    "start_row": index,
                    "start_y": row["y0"],
                    "heading": row["text"],
                    "page": page.get("page")
                })

                break

    for index, section in enumerate(sections):

        section["end_row"] = (
            sections[index + 1]["start_row"]
            if index + 1 < len(sections)
            else len(rows)
        )

        section["end_y"] = (
            sections[index + 1]["start_y"]
            if index + 1 < len(sections)
            else 1000
        )

    return sections


def get_section_rows(
    page,
    aliases,
    stop_aliases=None,
    max_rows=15
):

    rows = build_rows(page)

    stop_aliases = stop_aliases or []

    results = []

    for index, row in enumerate(rows):

        if not contains_known_label(
            row["text"],
            aliases
        ):
            continue

        section_rows = [row]

        for next_row in rows[
            index + 1:index + max_rows
        ]:

            if contains_known_label(
                next_row["text"],
                stop_aliases
            ):
                break

            section_rows.append(next_row)

        results.append({
            "page": page.get("page"),
            "heading": row["text"],
            "rows": section_rows,
            "text": "\n".join(
                item["text"]
                for item in section_rows
            ),
            "bbox": merge_boxes([
                item["bbox"]
                for item in section_rows
            ])
        })

    return results


# ==========================================
# 13. TABLE COLUMN EXTRACTION
# ==========================================

def extract_table_columns(
    page,
    column_aliases,
    max_data_rows=1,
    max_vertical_gap=35
):

    rows = build_rows(page)

    results = []

    for index, row in enumerate(rows):

        headers = []

        for field, aliases in column_aliases.items():

            matches = find_label_spans(
                row["words"],
                aliases
            )

            if not matches:
                continue

            match = matches[0]

            headers.append({
                "field": field,
                "label": match["label"],
                "bbox": match["bbox"],
                "x": center_x(match["bbox"])
            })

        if len(headers) < 2:
            continue

        headers.sort(
            key=lambda item: item["x"]
        )

        boundaries = [-1]

        for left, right in zip(
            headers,
            headers[1:]
        ):

            boundaries.append(
                (left["x"] + right["x"]) / 2
            )

        boundaries.append(1001)

        previous_bottom = row["y1"]

        for data_row in rows[
            index + 1:index + 1 + max_data_rows
        ]:

            gap = (
                data_row["y0"]
                - previous_bottom
            )

            if gap > max_vertical_gap:
                break

            values = {}

            for position, header in enumerate(headers):

                left = boundaries[position]

                right = boundaries[position + 1]

                cell_words = [
                    word
                    for word in data_row["words"]
                    if left <= center_x(
                        word["bbox"]
                    ) < right
                ]

                candidate = make_candidate(
                    page=page,
                    label=header["label"],
                    words=cell_words,
                    method="table_column",
                    priority=12,
                    evidence=(
                        row["text"]
                        + " | "
                        + data_row["text"]
                    )
                )

                if candidate:

                    candidate["field"] = header["field"]

                    values[header["field"]] = candidate

            if values:

                results.append(values)

            previous_bottom = data_row["y1"]

    return results


# ==========================================
# 14. DEBUG VISUAL ROWS AND CELLS
# ==========================================

def debug_layout(page):

    output = []

    for row in build_rows(page):

        output.append({
            "row_index": row["row_index"],
            "text": row["text"],
            "bbox": row["bbox"],
            "cells": [
                {
                    "text": cell["text"],
                    "bbox": cell["bbox"]
                }
                for cell in split_cells(row)
            ]
        })

    return output
