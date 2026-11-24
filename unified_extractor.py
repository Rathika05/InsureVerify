
"""
Unified Insurance Extraction Engine.

Responsibilities:
1. Insurance category detection
2. Coordinate-based field extraction
3. Native PDF table extraction
4. Additional health policy conditions
5. Candidate selection
6. Cross-field validation
7. Source page and bounding box verification

No Qwen, Ollama or external LLM.
"""

import re

from evidence_upgrade import extract_schedule, extract_owned_contacts
from datetime import datetime
from decimal import Decimal, InvalidOperation

from field_extractor import (
    TYPES,
    guess_type,
    get_schema,
    extract_fields as baseline_extract,
    _value_from,
    _find_box,
    _line_groups,
    CONTACT_FIELDS,
    CUSTOMER_CONTEXT,
    NON_CUSTOMER_CONTEXT,
    validate_field
)

from field_validators import (
    clean_value
)


# ==========================================
# 1. CONSTANTS
# ==========================================

HEALTH_CONDITIONS = {

    "9 Months Waiting Period Deleted":
        "Deletion of 9 months waiting period",

    "30 Days Waiting Period Deleted":
        "Deletion of 30 days waiting period",

    "2/4 Year Exclusion Deleted":
        "Deletion of 2/4 year exclusion",

    "Pre-existing Cover Opted":
        "Pre-existing cover opted"
}


DATE_FORMATS = (

    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%d-%B-%Y"
)


MONEY_FIELDS = {

    "Premium",
    "Gross Premium",
    "Net Premium",
    "Total Premium",
    "GST",
    "CGST",
    "SGST",
    "IGST",
    "Terrorism Premium",
    "Sum Insured / Assured",
    "Family Floater Cover",
    "Death Benefit",
    "IDV"
}


# ==========================================
# 2. BASIC HELPERS
# ==========================================

def missing(field):

    return {

        "field": field,
        "value": "",
        "page": None,
        "bbox": None,
        "status": "Not found",
        "evidence": ""
    }


def normalize(value):

    return re.sub(

        r"[^a-z0-9]+",

        " ",

        str(value or "").lower()

    ).strip()


def parse_date(value):

    value = clean_value(value)

    for fmt in DATE_FORMATS:

        try:

            return datetime.strptime(
                value,
                fmt
            )

        except ValueError:

            continue

    return None


def parse_money(value):

    if value is None:

        return None

    value = str(value).strip()

    value = re.sub(

        r"^(?:₹|Rs\.?|INR)\s*",

        "",

        value,

        flags=re.I
    )

    value = value.replace(",", "")

    try:

        amount = Decimal(value)

        return amount if amount.is_finite() else None

    except (InvalidOperation, ValueError):

        return None


def valid_bbox(box):

    if not isinstance(box, (list, tuple)):

        return False

    if len(box) != 4:

        return False

    try:

        x0, y0, x1, y1 = map(float, box)

        return (

            0 <= x0 < x1 <= 1000

            and

            0 <= y0 < y1 <= 1000
        )

    except (TypeError, ValueError):

        return False


def comparison_value(field, value):

    if field in MONEY_FIELDS:

        amount = parse_money(value)

        if amount is not None:

            return str(amount.normalize())

    if field in {

        "Policy Start",
        "Policy End",
        "Renewal Date",
        "Date of Birth"
    }:

        date = parse_date(value)

        if date:

            return date.strftime("%Y-%m-%d")

    return re.sub(

        r"\s+",

        "",

        str(value or "").lower()
    )


def mark_review(field, reason):

    if not field or not field.get("value"):

        return

    field["status"] = "Needs review"

    evidence = field.get("evidence", "")

    if reason not in evidence:

        field["evidence"] = (

            reason + " | " + evidence

        )[:500]


# ==========================================
# 3. FIELD VALUE VALIDATION
# ==========================================

def field_value_valid(field, value):

    value = clean_value(value)
    if re.match(r"^(Member|Rider) \d+ — ", field):
        field = field.split(" — ", 1)[1]

    if not value:

        return False

    if field in HEALTH_CONDITIONS:

        return value.upper() in {

            "YES",
            "NO"
        }

    if field in MONEY_FIELDS:

        amount = parse_money(value)

        if amount is None or amount < 0:

            return False

    return bool(

        validate_field(
            field,
            value
        )
    )


# ==========================================
# 4. WORDS INSIDE A TABLE CELL
# ==========================================

def words_inside_box(words, box):

    if not valid_bbox(box):

        return []

    x0, y0, x1, y1 = map(float, box)

    selected = []

    for word in words:

        word_box = word.get("bbox")

        if not valid_bbox(word_box):

            continue

        wx0, wy0, wx1, wy1 = map(

            float,

            word_box
        )

        center_x = (wx0 + wx1) / 2

        center_y = (wy0 + wy1) / 2

        if (

            x0 <= center_x <= x1

            and

            y0 <= center_y <= y1
        ):

            selected.append(word)

    if not selected:

        return []

    rows = _line_groups(selected)

    return [

        word

        for row in rows

        for word in row["words"]
    ]


# ==========================================
# 5. STORE FIELD CANDIDATES
# ==========================================

def add_candidate(

    found,

    page,

    field,

    value,

    evidence,

    bbox=None,

    verified=False,

    score=50
):

    value = clean_value(value)

    if not field_value_valid(field, value):

        return

    if not valid_bbox(bbox):

        bbox = None

    status = (

        "Found"

        if verified and bbox is not None

        else "Needs review"
    )

    item = {

        "field": field,

        "value": value,

        "page": page.get("page"),

        "bbox": list(bbox) if bbox else None,

        "status": status,

        "evidence": str(evidence or "")[:500],

        "_score": score
    }

    found.setdefault(

        field,

        []

    ).append(item)


# ==========================================
# 6. IMPORT UPDATED FIELD EXTRACTOR
# ==========================================

def import_baseline(payload, kind, found):

    results = baseline_extract(

        payload,

        kind
    )

    page_map = {

        page["page"]: page

        for page in payload.get("pages", [])
    }

    for result in results:

        field = result.get("field")

        value = result.get("value")

        if not field or not value:

            continue

        if result.get("status") == "Not found":

            continue

        page = page_map.get(

            result.get("page")
        )

        if page is None:

            continue

        evidence = result.get("evidence", "")

        if field in CONTACT_FIELDS:

            if NON_CUSTOMER_CONTEXT.search(evidence):

                continue

        add_candidate(

            found=found,

            page=page,

            field=field,

            value=value,

            evidence=evidence,

            bbox=result.get("bbox"),

            verified=(

                result.get("status") == "Found"
            ),

            score=120
        )


# ==========================================
# 7. TABLE LABEL MATCHING
# ==========================================

def match_table_label(text, schema):

    label = normalize(text)

    if not label:

        return None

    matches = set()

    for field, aliases in schema.items():

        for alias in aliases:

            if label == normalize(alias):

                matches.add(field)

    # Ambiguous labels must not be
    # assigned to an arbitrary field.

    if len(matches) != 1:

        return None

    return next(iter(matches))


# ==========================================
# 8. READ NATIVE PDF TABLE ROWS
# ==========================================

def get_table_rows(table):

    raw_rows = table.get("rows") or []

    grouped = {}

    for row_index, row in enumerate(raw_rows):

        if isinstance(row, dict):

            cells = row.get("cells")

            if cells is None:

                cells = [row]

        elif isinstance(row, list):

            cells = row

        else:

            continue

        for column_index, cell in enumerate(cells):

            if not isinstance(cell, dict):

                continue

            text = clean_value(

                cell.get("text", "")
            )

            if not text:

                continue

            try:

                r = int(

                    cell.get("row", row_index)
                )

                c = int(

                    cell.get("column", column_index)
                )

            except (TypeError, ValueError):

                continue

            grouped.setdefault(

                r,

                []

            ).append({

                "text": text,

                "column": c,

                "bbox": cell.get("bbox")
            })

    rows = []

    for row_number in sorted(grouped):

        cells = sorted(

            grouped[row_number],

            key=lambda cell: cell["column"]
        )

        rows.append(cells)

    return rows


# ==========================================
# 9. CUSTOMER OWNERSHIP IN TABLES
# ==========================================

def table_customer_verified(

    field,

    label,

    evidence
):

    if field not in CONTACT_FIELDS:

        return True

    if NON_CUSTOMER_CONTEXT.search(evidence):

        return False

    return bool(

        CUSTOMER_CONTEXT.search(label)
    )


# ==========================================
# 10. STORE TABLE CELL VALUE
# ==========================================

def add_table_value(

    page,

    found,

    field,

    label,

    cell,

    evidence,

    verified=True
):

    raw = cell.get("text", "")

    value = _value_from(

        raw,

        field
    )

    if not value:

        return

    if field in CONTACT_FIELDS:

        if not table_customer_verified(

            field,

            label,

            evidence
        ):

            return

    cell_words = words_inside_box(

        page.get("words") or [],

        cell.get("bbox")
    )

    if not cell_words:

        return

    bbox = _find_box(

        cell_words,

        value
    )

    if not valid_bbox(bbox):

        return

    add_candidate(

        found=found,

        page=page,

        field=field,

        value=value,

        evidence=evidence,

        bbox=bbox,

        verified=verified,

        score=135
    )


# ==========================================
# 11. NATIVE KEY-VALUE TABLE EXTRACTION
# ==========================================

def extract_key_value_table(

    page,

    rows,

    schema,

    found
):

    for cells in rows:

        for index, cell in enumerate(cells):

            label = cell["text"]

            field = match_table_label(

                label,

                schema
            )

            if not field:

                continue

            if index + 1 >= len(cells):

                continue

            value_cell = cells[index + 1]

            # Only use the adjacent column.

            if (

                value_cell["column"]

                != cell["column"] + 1
            ):

                continue

            # The next cell must be a value,
            # not another known field label.

            if match_table_label(

                value_cell["text"],

                schema
            ):

                continue

            evidence = " | ".join(

                item["text"]

                for item in cells
            )

            add_table_value(

                page=page,

                found=found,

                field=field,

                label=label,

                cell=value_cell,

                evidence=evidence,

                verified=True
            )


# ==========================================
# 12. NATIVE COLUMN TABLE EXTRACTION
# ==========================================

def extract_column_table(

    page,

    rows,

    schema,

    found
):

    for row_index, header in enumerate(rows):

        if row_index + 1 >= len(rows):

            continue

        columns = {}

        for cell in header:

            field = match_table_label(

                cell["text"],

                schema
            )

            if field:

                columns[cell["column"]] = {

                    "field": field,

                    "label": cell["text"]
                }

        if len(columns) < 2:

            continue

        # A header must contain field labels,
        # not mixed label-value cells.

        if len(columns) != len(header):

            continue

        for data_row in rows[

            row_index + 1:row_index + 4
        ]:

            if any(

                match_table_label(

                    cell["text"],

                    schema
                )

                for cell in data_row
            ):

                break

            evidence = (

                " | ".join(

                    cell["text"]

                    for cell in header
                )

                + " || "

                + " | ".join(

                    cell["text"]

                    for cell in data_row
                )
            )

            for cell in data_row:

                column = columns.get(

                    cell["column"]
                )

                if not column:

                    continue

                add_table_value(

                    page=page,

                    found=found,

                    field=column["field"],

                    label=column["label"],

                    cell=cell,

                    evidence=evidence,

                    # Multiple data rows may represent
                    # different insured members.

                    verified=False
                )


# ==========================================
# 13. NATIVE TABLE COORDINATION
# ==========================================

def extract_native_tables(

    page,

    schema,

    found
):

    tables = page.get("tables") or []

    for table in tables:

        if not isinstance(table, dict):

            continue

        rows = get_table_rows(table)

        if not rows:

            continue

        extract_key_value_table(

            page,

            rows,

            schema,

            found
        )

        extract_column_table(

            page,

            rows,

            schema,

            found
        )


# ==========================================
# 14. PREMIUM / GST COLUMN TABLE
# ==========================================

def extract_premium_columns(page, found):

    rows = _line_groups(

        page.get("words") or []
    )

    for index, header in enumerate(rows):

        if index + 1 >= len(rows):

            continue

        if normalize(header["text"]) != "premium gst total":

            continue

        data = rows[index + 1]

        distance = data["y"] - header["y"]

        if not 0 < distance <= 30:

            continue

        pattern = (

            r"(?<![\w.,])"

            r"(?:₹|Rs\.?|INR)?\s*"

            r"(\d[\d,]*(?:\.\d{1,2})?)"

            r"(?![\w.,])"
        )

        matches = list(

            re.finditer(

                pattern,

                data["text"],

                re.I
            )
        )

        if len(matches) != 3:

            continue

        values = [

            match.group(1)

            for match in matches
        ]

        amounts = [

            parse_money(value)

            for value in values
        ]

        if any(

            amount is None

            for amount in amounts
        ):

            continue

        if abs(

            amounts[0]

            + amounts[1]

            - amounts[2]

        ) > Decimal("1.00"):

            continue

        fields = [

            "Premium",

            "GST",

            "Total Premium"
        ]

        for field, value, match in zip(

            fields,

            values,

            matches
        ):

            value_words = words_in_span(

                data,

                match.start(1),

                match.end(1)
            )

            bbox = _find_box(

                value_words,

                value
            )

            if not valid_bbox(bbox):

                continue

            add_candidate(

                found=found,

                page=page,

                field=field,

                value=value,

                evidence=(

                    header["text"]

                    + " | "

                    + data["text"]
                ),

                bbox=bbox,

                verified=False,

                score=110
            )


# ==========================================
# 15. WORDS IN A TEXT SPAN
# ==========================================

def words_in_span(row, start, end):

    selected = []

    cursor = 0

    for word in row["words"]:

        text = str(word.get("text", ""))

        word_start = cursor

        word_end = cursor + len(text)

        if word_start < end and word_end > start:

            selected.append(word)

        cursor = word_end + 1

    return selected


# ==========================================
# 16. HEALTH POLICY CONDITIONS
# ==========================================

def extract_health_conditions(page, found):

    rows = _line_groups(

        page.get("words") or []
    )

    for row in rows:

        text = row["text"]

        for field, label in HEALTH_CONDITIONS.items():

            label_parts = label.split()

            pattern = (

                r"\b"

                + r"\s+".join(

                    re.escape(part)

                    for part in label_parts
                )

                + r"\s*[:=-]?\s*(YES|NO)\b"
            )

            match = re.search(

                pattern,

                text,

                re.I
            )

            if not match:

                continue

            value = match.group(1)

            selected_words = words_in_span(

                row,

                match.start(1),

                match.end(1)
            )

            bbox = _find_box(

                selected_words,

                value
            )

            if not valid_bbox(bbox):

                continue

            add_candidate(

                found=found,

                page=page,

                field=field,

                value=value,

                evidence=text,

                bbox=bbox,

                verified=True,

                score=130
            )


# ==========================================
# 17. SELECT BEST FIELD CANDIDATE
# ==========================================

def select_best(field, candidates):

    if not candidates:

        return missing(field)

    valid = [

        candidate

        for candidate in candidates

        if field_value_valid(

            field,

            candidate.get("value")
        )
    ]

    if not valid:

        return missing(field)

    unique = {}

    for candidate in valid:

        key = (

            comparison_value(

                field,

                candidate["value"]
            ),

            candidate.get("page"),

            tuple(

                candidate.get("bbox") or []
            )
        )

        old = unique.get(key)

        if old is None or (

            candidate["status"] == "Found",

            candidate["_score"]

        ) > (

            old["status"] == "Found",

            old["_score"]
        ):

            unique[key] = candidate

    valid = list(unique.values())

    valid.sort(

        key=lambda item: (

            item["status"] != "Found",

            not valid_bbox(item.get("bbox")),

            -item["_score"],

            item.get("page") or 9999
        )
    )

    best = valid[0].copy()

    verified_values = {

        comparison_value(

            field,

            item["value"]
        )

        for item in valid

        if item["status"] == "Found" and item["_score"] >= best["_score"] - 10
    }

    if len(verified_values) > 1:

        mark_review(

            best,

            "Multiple different values detected. "
            "Verify the source document."
        )

    best.pop("_score", None)

    return best


# ==========================================
# 18. POLICY DATE VALIDATION
# ==========================================

def validate_policy_dates(field_map):

    start = field_map.get("Policy Start")

    end = field_map.get("Policy End")

    if start and end:

        start_date = parse_date(

            start.get("value", "")
        )

        end_date = parse_date(

            end.get("value", "")
        )

        if start_date and end_date:

            if start_date > end_date:

                reason = (

                    "Policy start date is later "
                    "than policy end date."
                )

                mark_review(start, reason)

                mark_review(end, reason)

    dob = field_map.get("Date of Birth")

    if dob and dob.get("value"):

        birth_date = parse_date(

            dob["value"]
        )

        if birth_date:

            if birth_date > datetime.now():

                mark_review(

                    dob,

                    "Date of birth is in the future."
                )


# ==========================================
# 19. PREMIUM CONSISTENCY VALIDATION
# ==========================================

def validate_premium_values(field_map):

    total = field_map.get("Total Premium")

    if not total or not total.get("value"):

        return

    total_amount = parse_money(

        total["value"]
    )

    if total_amount is None:

        return

    # Do not assume Gross Premium,
    # Net Premium and Total Premium
    # mean the same thing.

    # Insurance policies can include:
    # taxes, discounts, loading,
    # terrorism premium and other charges.

    # Never calculate a missing field
    # and present it as an extracted value.

    # Check a simple Premium + GST table
    # only when the source explicitly
    # contains that exact structure.

    premium = field_map.get("Premium")

    gst = field_map.get("GST")

    if not premium or not gst:

        return

    if not premium.get("value"):

        return

    if not gst.get("value"):

        return

    if not (

        premium.get("page")

        == gst.get("page")

        == total.get("page")
    ):

        return

    evidence = total.get("evidence", "")

    header = evidence.split("|")[0]

    if normalize(header) != "premium gst total":

        return

    premium_amount = parse_money(

        premium["value"]
    )

    gst_amount = parse_money(

        gst["value"]
    )

    if premium_amount is None:

        return

    if gst_amount is None:

        return

    difference = abs(

        premium_amount

        + gst_amount

        - total_amount
    )

    if difference > Decimal("1.00"):

        reason = (

            "Premium and GST do not match "
            "the printed total premium."
        )

        mark_review(premium, reason)

        mark_review(gst, reason)

        mark_review(total, reason)


# ==========================================
# 20. FINAL SOURCE VERIFICATION
# ==========================================

def verify_sources(fields, payload):

    page_map = {

        page["page"]: page

        for page in payload.get("pages", [])
    }

    for field in fields:

        name = field["field"]

        value = field.get("value")

        if not value:

            field.update(missing(name))

            continue

        if not field_value_valid(name, value):

            field.update(missing(name))

            continue

        page = page_map.get(

            field.get("page")
        )

        if page is None:

            mark_review(

                field,

                "Source page could not be verified."
            )

            field["page"] = None

            field["bbox"] = None

            continue

        bbox = field.get("bbox")

        if not valid_bbox(bbox):

            field["bbox"] = None

            mark_review(

                field,

                "Source bounding box unavailable."
            )

            continue

        source_words = words_inside_box(

            page.get("words") or [],

            bbox
        )

        if not source_words:

            field["bbox"] = None

            mark_review(

                field,

                "No source words inside the bounding box."
            )

            continue

        source_box = _find_box(

            source_words,

            value
        )

        if not valid_bbox(source_box):

            mark_review(

                field,

                "Exact value could not be verified "
                "inside the source region."
            )

            continue

        field["bbox"] = list(source_box)

    return fields


# ==========================================
# 21. FINAL CROSS-FIELD VALIDATION
# ==========================================

def cross_validate(fields, payload):

    fields = verify_sources(

        fields,

        payload
    )

    field_map = {

        field["field"]: field

        for field in fields
    }

    validate_policy_dates(field_map)

    validate_premium_values(field_map)

    return fields


# ==========================================
# 22. MAIN EXTRACTION ENTRY POINT
# ==========================================

def extract_fields(payload, kind="Auto-detect"):

    pages = payload.get("pages") or []

    # STEP 1:
    # Detect insurance category.

    if kind == "Auto-detect":

        full_text = "\n".join(

            str(page.get("text", ""))

            for page in pages
        )

        kind = guess_type(full_text)

    if kind not in TYPES:

        kind = "Others"

    # STEP 2:
    # Build the required field schema.

    schema = get_schema(

        payload,

        kind
    )

    found = {}

    # STEP 3:
    # Run updated field_extractor.py.

    import_baseline(

        payload,

        kind,

        found
    )

    # STEP 4:
    # Extract additional native table values.

    for page in pages:

        if not page.get("words"):

            continue

        for field, value, bbox, evidence, score in list(extract_schedule(page, kind)) + list(extract_owned_contacts(page)):
            if field in schema:
                add_candidate(found, page, field, value, evidence, bbox, True, score)

        extract_native_tables(

            page,

            schema,

            found
        )

        extract_premium_columns(

            page,

            found
        )

        # STEP 5:
        # Extract health-specific conditions.

        if kind in {

            "Health Insurance",

            "Family Insurance"
        }:

            extract_health_conditions(

                page,

                found
            )

    # STEP 6:
    # Select the best candidate for each field.

    result = [

        select_best(

            field,

            found.get(field, [])
        )

        for field in schema
    ]

    # Preserve actual member/rider rows, not unrelated schema fields.
    for field, candidates in found.items():
        if re.match(r"^(Member|Rider) \d+ — ", field):
            item = select_best(field, candidates)
            if item.get("value"):
                result.append(item)

    # STEP 7:
    # Verify source and cross-field consistency.

    result = cross_validate(

        result,

        payload
    )

    for page in pages:
        page.pop("_cached_spatial_rows", None)
    return result

# Dynamic mode: keep verified known values and discover additional printed labels.
from dynamic_extractor import discover
_extract_known = extract_fields

def extract_fields(payload, kind='Auto-detect'):
    resolved = guess_type(payload) if kind == 'Auto-detect' else kind
    schema = get_schema(payload, resolved)
    known = _extract_known(payload, kind)
    result = [item for item in known if item.get('value') and item.get('status') != 'Not found']
    result.extend(discover(payload, schema))
    for page in payload.get('pages', []):
        page.pop('_cached_spatial_rows', None)
    return result
