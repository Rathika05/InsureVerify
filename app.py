
from pathlib import Path

import io
import json
import tempfile
import zipfile
import hashlib
import html

import streamlit as st
import fitz

from PIL import Image, ImageDraw

from layoutlmv3_pipeline import (
    load_model,
    process_pdf
)

from field_extractor import (
    TYPES,
    guess_type
)

from unified_extractor import (
    extract_fields as unified_extract_fields
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="InsureVerify | Insurance Document Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM UI STYLES
# ==========================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f7f9fc;
        color: #172b4d;
    }

    [data-testid="stSidebar"] {
        background: #edf3fb;
    }

    .block-container {
        padding-top: 1.2rem;
        max-width: 1800px;
    }

    h1, h2, h3, h4 {
        color: #172b4d;
        letter-spacing: -0.02em;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #d7e3f3 !important;
        background: #ffffff;
        border-radius: 12px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] p {
        overflow-wrap: anywhere;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d7e3f3;
        border-radius: 12px;
        padding: 12px;
    }

    .stButton button {
        border-radius: 9px;
        min-height: 2.25rem;
    }

    .header {
        background: linear-gradient(
            105deg,
            #e8f1ff,
            #f7fbff
        );
        border: 1px solid #d7e3f3;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }

    .header h2 {
        margin: 0;
        color: #12346a;
    }

    .header p {
        margin: 6px 0 0;
        color: #526785;
    }

    .field-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #526785;
        margin-bottom: 8px;
    }

    .field-value {
        font-size: 1rem;
        font-weight: 600;
        color: #172b4d;
        min-height: 2.8rem;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }

    .missing-value {
        color: #94a3b8;
    }

    .group-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #12346a;
        border-bottom: 1px solid #d7e3f3;
        padding-bottom: 8px;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    .status-found {
        color: #15803d;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .status-review {
        color: #b45309;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .status-missing {
        color: #dc2626;
        font-size: 0.78rem;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 3. FIELD DISPLAY NAMES
# ==========================================

DISPLAY_NAMES = {
    "Insurer": "Insurance Company",
    "Policyholder": "Policyholder Name",
    "Proposer Name": "Proposer Name",
    "Insured Person": "Insured Person Name",
    "Primary Insured": "Primary Insured Name",
    "Life Assured": "Life Assured Name",
    "Mobile Number": "Mobile Number",
    "Phone Number": "Phone / Contact Number",
    "Gross Premium": "Gross Premium",
    "Net Premium": "Net Premium",
    "CGST": "CGST Amount",
    "SGST": "SGST Amount",
    "IGST": "IGST Amount",
    "Terrorism Premium": "Terrorism Premium",
    "Email ID": "Email ID",
    "Residential Address": "Residential Address",
    "Date of Birth": "Date of Birth",
    "Customer ID": "Customer ID",
    "Policy Start": "Policy Start Date",
    "Policy End": "Policy End Date",
    "CC": "Engine Capacity (CC)",
    "IDV": "Insured Declared Value (IDV)",
    "GST": "GST Amount",
    "Hospital / TPA": "Hospital / TPA Name"
}


# ==========================================
# 4. FIELD GROUPS
# ==========================================

GROUPS = {

    "Record": [
        "Insurer"
    ],

    "Policy & Customer": [
        "Policy Number",
        "Policyholder",
        "Proposer Name",
        "Insured Person",
        "Primary Insured",
        "Life Assured",
        "Customer ID",
        "Mobile Number",
        "Phone Number",
        "Email ID",
        "Date of Birth",
        "Gender",
        "Age",
        "Residential Address",
        "Pincode"
    ],

    "Vehicle Details": [
        "Registration Number",
        "Vehicle Make",
        "Vehicle Model",
        "Engine Number",
        "Chassis Number",
        "Manufacturing Year",
        "Registration Year",
        "Seating Capacity",
        "Seats",
        "Body Type",
        "CC",
        "IDV",
        "GVW"
    ],

    "Policy Period": [
        "Policy Start",
        "Policy End",
        "Renewal Date"
    ],

    "Premium & Coverage": [
        "Premium",
        "Gross Premium",
        "Net Premium",
        "Terrorism Premium",
        "GST",
        "CGST",
        "SGST",
        "IGST",
        "Total Premium",
        "Sum Insured / Assured",
        "Family Floater Cover",
        "Death Benefit",
        "Premium Frequency"
    ],

    "Property Insurance Details": [
        "Bank / Financial Institution",
        "Location of Risk",
        "Occupation"
    ],

    "Health & TPA": [
        "Waiting Period",
        "Co-payment",
        "Hospital / TPA",
        "Doctor",
        "9 Months Waiting Period Deleted",
        "30 Days Waiting Period Deleted",
        "2/4 Year Exclusion Deleted",
        "Pre-existing Cover Opted"
    ],

    "Family & Nominee": [
        "Family Member",
        "Relationship",
        "Nominee",
        "Number of Members"
    ]
}


# ==========================================
# 5. CATEGORY SPECIFIC GROUPS
# ==========================================

KIND_GROUPS = {

    "Vehicle Insurance": [
        "Record",
        "Policy & Customer",
        "Vehicle Details",
        "Policy Period",
        "Premium & Coverage"
    ],

    "Health Insurance": [
        "Record",
        "Policy & Customer",
        "Policy Period",
        "Premium & Coverage",
        "Health & TPA"
    ],

    "Family Insurance": [
        "Record",
        "Policy & Customer",
        "Policy Period",
        "Premium & Coverage",
        "Health & TPA",
        "Family & Nominee"
    ],

    "Life Insurance": [
        "Record",
        "Policy & Customer",
        "Policy Period",
        "Premium & Coverage",
        "Family & Nominee"
    ],

    "Others": [
        "Record",
        "Policy & Customer",
        "Policy Period",
        "Premium & Coverage",
        "Property Insurance Details"
    ]
}


# ==========================================
# 6. LOAD MODEL
# ==========================================

@st.cache_resource(
    show_spinner="Loading pretrained LayoutLMv3..."
)
def model_resources():

    local = (
        Path(__file__).parent
        / "saved_layoutlmv3"
    )

    return load_model(
        local if local.is_dir() else None
    )


# ==========================================
# 7. SESSION STATE
# ==========================================

run_encoder = st.sidebar.checkbox("Run pretrained LayoutLMv3 feature encoder", value=False, help="Optional: base encoder features do not determine extracted fields.")

DEFAULT_STATE = {
    "pdf_bytes": None,
    "pdf_name": None,
    "pdf_hash": None,
    "payload": None,
    "fields": [],
    "detected": "Others",
    "kind": "Others",
    "processed_choice": None,
    "current_page": 1,
    "active_field": None,
    "output_zip": None
}

for key, default in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = default


def clear_active_result():

    for key, default in DEFAULT_STATE.items():

        st.session_state[key] = default


def clear_results():
    clear_active_result()
    st.session_state.batch_results = {}
    st.session_state.batch_errors = {}
    st.session_state.pop("selected_document", None)
    st.session_state.active_document = None


for key, default in {
    "batch_results": {}, "batch_errors": {}, "active_document": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================================
# 8. PDF HASH
# ==========================================

def document_hash(pdf_bytes):

    return hashlib.sha256(
        pdf_bytes
    ).hexdigest()


# ==========================================
# 9. BOUNDING BOX VALIDATION
# ==========================================

def valid_bbox(box):

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


# ==========================================
# 10. PDF PAGE RENDERING
# ==========================================

def render_page(
    pdf_bytes,
    page_number,
    box=None,
    scale=1.5
):

    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as doc:

        page_number = max(
            1,
            min(page_number, len(doc))
        )

        page = doc[page_number - 1]

        pix = page.get_pixmap(
            matrix=fitz.Matrix(
                scale,
                scale
            ),
            colorspace=fitz.csRGB,
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            (pix.width, pix.height),
            pix.samples
        )

        if valid_bbox(box):

            x0, y0, x1, y1 = map(float, box)

            padding = 4

            rect = (
                max(
                    0,
                    int(x0 / 1000 * image.width)
                    - padding
                ),

                max(
                    0,
                    int(y0 / 1000 * image.height)
                    - padding
                ),

                min(
                    image.width - 1,
                    int(x1 / 1000 * image.width)
                    + padding
                ),

                min(
                    image.height - 1,
                    int(y1 / 1000 * image.height)
                    + padding
                )
            )

            overlay = Image.new(
                "RGBA",
                image.size,
                (0, 0, 0, 0)
            )

            draw = ImageDraw.Draw(
                overlay
            )

            draw.rectangle(
                rect,
                fill=(255, 197, 42, 90),
                outline=(255, 147, 0, 255),
                width=4
            )

            image = Image.alpha_composite(
                image.convert("RGBA"),
                overlay
            ).convert("RGB")

    return image


# ==========================================
# 11. LOCATE FIELD
# ==========================================

def jump(field):

    page = field.get("page")

    payload = st.session_state.payload

    if not page or not payload or not valid_bbox(field.get("bbox")):

        return

    if not 1 <= page <= payload["processed_pages"]:

        return

    st.session_state.current_page = page

    st.session_state.active_field = field["field"]


def previous_page():

    st.session_state.current_page = max(
        1,
        st.session_state.current_page - 1
    )

    st.session_state.active_field = None


def next_page():

    total = st.session_state.payload[
        "processed_pages"
    ]

    st.session_state.current_page = min(
        total,
        st.session_state.current_page + 1
    )

    st.session_state.active_field = None


# ==========================================
# 12. DOWNLOAD ZIP
# ==========================================

def make_zip(payload, fields):

    memory = io.BytesIO()

    with zipfile.ZipFile(
        memory,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        archive.writestr(
            "extracted_fields.json",
            json.dumps(
                fields,
                indent=2,
                ensure_ascii=False
            )
        )

        archive.writestr(
            "layoutlmv3_extraction.json",
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False
            )
        )

        full_text = "\n\n".join(
            (
                f'===== PAGE {page["page"]} =====\n'
                f'{page["text"]}'
            )
            for page in payload["pages"]
        )

        archive.writestr(
            "full_text.txt",
            full_text
        )

        tables = [
            {
                "page": page["page"],
                "tables": page.get("tables", [])
            }
            for page in payload["pages"]
        ]

        archive.writestr(
            "detected_tables.json",
            json.dumps(
                tables,
                indent=2,
                ensure_ascii=False
            )
        )

    return memory.getvalue()


# ==========================================
# 13. GROUP EXTRACTED FIELDS
# ==========================================

def group_fields(fields, kind):

    group_names = KIND_GROUPS.get(
        kind,
        KIND_GROUPS["Others"]
    )

    grouped = {
        name: []
        for name in group_names
    }

    grouped["Other Extracted Fields"] = []

    field_map = {
        field["field"]: field
        for field in fields
    }

    used = set()

    # Maintain the desired UI field order.
    for group_name in group_names:

        for field_name in GROUPS.get(
            group_name,
            []
        ):

            if field_name in field_map:

                grouped[group_name].append(
                    field_map[field_name]
                )

                used.add(field_name)

    # Preserve any additional extracted fields.
    for field in fields:

        if field["field"] not in used:

            grouped[
                "Other Extracted Fields"
            ].append(field)

    return grouped


# ==========================================
# 14. FIELD CARD COMPONENT
# ==========================================

def render_field_card(field, key):

    field_name = str(
        field.get(
            "field",
            "Unknown field"
        )
    )

    display_name = DISPLAY_NAMES.get(
        field_name,
        field_name
    )

    raw_value = field.get("value")

    has_value = (
        raw_value is not None
        and str(raw_value).strip() != ""
    )

    value = (
        str(raw_value)
        if has_value
        else "—"
    )

    status = field.get(
        "status",
        "Not found"
    )

    source_page = field.get("page")

    active = (
        st.session_state.active_field
        == field_name
    )

    with st.container(border=True):

        label = html.escape(
            display_name
        )

        if active:

            label = "📍 " + label

        st.markdown(
            f'<div class="field-label">{label}</div>',
            unsafe_allow_html=True
        )

        value_class = (
            "field-value"
            if has_value
            else "field-value missing-value"
        )

        st.markdown(
            f'<div class="{value_class}">'
            f'{html.escape(value)}'
            f'</div>',
            unsafe_allow_html=True
        )

        if status == "Found":

            st.markdown(
                '<div class="status-found">'
                f'● Found · Page {source_page}'
                '</div>',
                unsafe_allow_html=True
            )

        elif status == "Needs review":

            page_label = (
                source_page
                if source_page
                else "Unknown"
            )

            st.markdown(
                '<div class="status-review">'
                f'● Needs review · Page {page_label}'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="status-missing">'
                '● Not found'
                '</div>',
                unsafe_allow_html=True
            )

        st.button(
            "🔎 Locate",
            key=f"locate_{key}",
            disabled=(source_page is None or not valid_bbox(field.get("bbox"))),
            use_container_width=True,
            on_click=jump,
            args=(field,)
        )

        if field.get("evidence"):

            with st.expander(
                "Source evidence"
            ):

                st.text(
                    field["evidence"]
                )


# ==========================================
# 15. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🛡️ InsureVerify")

    st.caption(
        "DOCUMENT EXTRACTION & "
        "EVIDENCE VERIFICATION"
    )

    st.divider()

    st.markdown(
        "### Insurance category"
    )

    choice = st.selectbox(
        "Choose insurance type",
        ["Auto-detect"] + TYPES,
        help=(
            "Auto-detect uses document keywords. "
            "You can manually select a category."
        )
    )

    st.markdown(
        "### Upload documents"
    )

    uploads = st.file_uploader(
        "Insurance PDFs", type=["pdf"], accept_multiple_files=True
    )
    documents = {
        f"{item.name} | {document_hash(item.getvalue())}": item
        for item in uploads
    }

    st.divider()

    extract = st.button(
        "⚡ Extract All PDFs",
        type="primary",
        use_container_width=True,
        disabled=not uploads
    )

    st.button(
        "Clear results",
        use_container_width=True,
        on_click=clear_results,
        disabled=(
            not st.session_state.batch_results and not st.session_state.batch_errors
        )
    )

    st.divider()

    st.markdown(
        "### Extraction technology"
    )

    st.caption(
        "PyMuPDF · Tesseract OCR · "
        "LayoutLMv3-base · Spatial rules · "
        "Field validation"
    )

    st.caption(
        "No Qwen or Ollama. "
        "LayoutLMv3-base produces features, "
        "not trained insurance field labels."
    )


# ==========================================
# 16. PAGE HEADER
# ==========================================

st.markdown(
    """
<div class="header">
<h2>Insurance Document Intelligence</h2>
<p>Extract policy and customer details, inspect the original PDF, and verify supporting evidence.</p>
</div>
""",
    unsafe_allow_html=True
)


# ==========================================
# 17. PROCESS UPLOADED PDF
# ==========================================

def process_document(upload, choice, resources):
    raw = upload.getvalue()
    st.write(
        "Checking PDF document..."
    )

    with fitz.open(
        stream=raw,
        filetype="pdf"
    ) as document:

        if document.needs_pass:

            raise RuntimeError(
                "Password-protected PDFs "
                "are not supported."
            )

        if len(document) == 0:

            raise RuntimeError(
                "The PDF contains no pages."
            )

    st.write("Preparing extraction...")

    processor, model, device = resources

    with tempfile.TemporaryDirectory() as td:

        target = (
            Path(td)
            / Path(upload.name).name
        )

        target.write_bytes(raw)

        st.write(
            "Extracting PDF text and "
            "word coordinates..."
        )

        st.write(
            "Detecting available tables "
            "and optional model inference..."
        )

        payload, _ = process_pdf(
            target,
            Path(td) / "results",
            processor,
            model,
            device
        )

    full_text = "\n".join(
        page["text"]
        for page in payload["pages"]
    )

    detected = guess_type(
        full_text
    )

    kind = (
        detected
        if choice == "Auto-detect"
        else choice
    )

    st.write(
        f"Insurance category: {kind}"
    )

    st.write(
        "Extracting policyholder, "
        "customer and insurance fields..."
    )

    fields = unified_extract_fields(
        payload,
        kind
    )

    st.write(
        "Preparing validated fields "
        "and source evidence..."
    )

    output_zip = make_zip(
        payload,
        fields
    )

    return dict(pdf_bytes=raw, pdf_name=upload.name,
                pdf_hash=document_hash(raw), payload=payload,
                detected=detected, kind=kind, processed_choice=choice,
                fields=fields, current_page=1, active_field=None,
                output_zip=output_zip)


if extract and documents:
    clear_results()
    resources = (None, None, "cpu")
    try:
        if run_encoder:
            resources = model_resources()
    except Exception as error:
        st.error(f"Model loading failed: {error}")
        st.stop()
    bar = st.progress(0, text="Processing PDFs...")
    for index, (doc_id, upload) in enumerate(documents.items(), 1):
        with st.status(f"{index}/{len(documents)}: {upload.name}", expanded=False) as progress:
            try:
                result = process_document(upload, choice, resources)
                st.session_state.batch_results[doc_id] = result
                progress.update(label=f"Completed: {upload.name}", state="complete")
            except Exception as error:
                st.session_state.batch_errors[doc_id] = str(error)
                progress.update(label=f"Failed: {upload.name}", state="error")
        bar.progress(index / len(documents), text=f"Processed {index}/{len(documents)} PDFs")

# Only display results for currently uploaded documents.
for store in (st.session_state.batch_results, st.session_state.batch_errors):
    for doc_id in list(store):
        if doc_id not in documents:
            del store[doc_id]

results = st.session_state.batch_results
errors = st.session_state.batch_errors
if documents:
    st.caption(f"Uploaded: {len(documents)} | Completed: {len(results)} | Failed: {len(errors)} | Pending: {len(documents)-len(results)-len(errors)}")
for doc_id, error in errors.items():
    st.error(f"{documents[doc_id].name}: {error}")
if results:
    options = list(results)
    if st.session_state.get("selected_document") not in options:
        st.session_state.selected_document = options[0]
    selected = st.selectbox("Choose PDF to view", options,
        format_func=lambda k: f"{results[k]['pdf_name']} — {results[k]['kind']} [{results[k]['pdf_hash'][:8]}]",
        key="selected_document")
    if selected != st.session_state.active_document:
        st.session_state.update(results[selected])
        st.session_state.active_document = selected
        for widget in ("text_page", "table_page", "model_page"):
            st.session_state.pop(widget, None)
else:
    clear_active_result()
    st.session_state.active_document = None

# ==========================================
# 18. DISPLAY EXTRACTION RESULTS
# ==========================================

if st.session_state.payload:

    payload = st.session_state.payload

    fields = st.session_state.fields

    kind = st.session_state.kind

    if choice != st.session_state.processed_choice:

        st.info(
            "Insurance category selection changed. "
            "Click Extract & verify to refresh."
        )


    # ======================================
    # 19. SUMMARY METRICS
    # ======================================

    found_count = sum(
        field["status"] == "Found"
        for field in fields
    )

    review_count = sum(
        field["status"] == "Needs review"
        for field in fields
    )

    missing_count = sum(
        field["status"] == "Not found"
        for field in fields
    )

    total_fields = len(fields)

    located_percentage = (
        found_count / total_fields
        if total_fields
        else 0
    )

    a, b, c, d, e = st.columns(5)

    a.metric(
        "Insurance type",
        kind
    )

    b.metric(
        "PDF pages",
        payload["processed_pages"]
    )

    c.metric(
        "Found",
        found_count
    )

    d.metric(
        "Needs review",
        review_count
    )

    e.metric(
        "Unresolved / absent",
        missing_count
    )

    st.progress(
        located_percentage
    )

    st.caption(
        f"{found_count} of {total_fields} "
        "fields marked Found. "
        "Configured schema coverage, not extraction accuracy. "
        "Unresolved / absent includes optional fields that may not exist in this PDF."
    )

    st.caption(
        f'📄 {st.session_state.pdf_name}'
        f' · Suggested category: '
        f'{st.session_state.detected}'
        f' · Model: {payload.get("model", "")}'
    )

    st.divider()


    # ======================================
    # 20. PDF VIEWER AND FIELD DETAILS
    # ======================================

    viewer, details = st.columns(
        [1.05, 1],
        gap="large"
    )


    # ======================================
    # 21. LEFT SIDE - ORIGINAL PDF
    # ======================================

    with viewer:

        st.subheader(
            "📑 Original PDF & evidence"
        )

        total_pages = payload[
            "processed_pages"
        ]

        p1, p2, p3 = st.columns(
            [1, 2, 1]
        )

        with p1:

            st.button(
                "◀ Previous",
                use_container_width=True,
                disabled=(
                    st.session_state.current_page <= 1
                ),
                on_click=previous_page
            )

        with p2:

            st.markdown(
                '<div style="text-align:center;'
                'padding:8px">'
                f'Page {st.session_state.current_page}'
                f' of {total_pages}'
                '</div>',
                unsafe_allow_html=True
            )

        with p3:

            st.button(
                "Next ▶",
                use_container_width=True,
                disabled=(
                    st.session_state.current_page
                    >= total_pages
                ),
                on_click=next_page
            )

        zoom = st.slider(
            "PDF render quality",
            min_value=1.0,
            max_value=2.5,
            value=1.5,
            step=0.25
        )

        active = next(
            (
                field
                for field in fields
                if (
                    field["field"]
                    == st.session_state.active_field

                    and field.get("page")
                    == st.session_state.current_page
                )
            ),
            None
        )

        active_box = (
            active.get("bbox")
            if active
            else None
        )

        with st.container(
            height=850,
            border=True
        ):

            st.image(
                render_page(
                    st.session_state.pdf_bytes,
                    st.session_state.current_page,
                    active_box,
                    scale=zoom
                ),
                use_container_width=True
            )

        if active:

            name = DISPLAY_NAMES.get(
                active["field"],
                active["field"]
            )

            if valid_bbox(
                active.get("bbox")
            ):

                if active["status"] == "Found":

                    st.success(
                        f'Highlighted: {name} — '
                        f'{active["value"]}'
                    )

                else:

                    st.warning(
                        f'Candidate highlighted: {name} — '
                        f'{active["value"]}. '
                        'Manual verification required.'
                    )

            else:

                st.warning(
                    "Source page opened, but exact "
                    "coordinates could not be verified."
                )


    # ======================================
    # 22. RIGHT SIDE - EXTRACTED FIELDS
    # ======================================

    with details:

        st.subheader(
            "📋 Extracted policy fields"
        )

        st.caption(
            "Click Locate to inspect the "
            "supporting PDF evidence."
        )

        status_filter = st.selectbox(
            "Filter fields",
            [
                "Extracted fields",
                "All fields",
                "Found",
                "Needs review",
                "Not found"
            ]
        )

        grouped = group_fields(
            fields,
            kind
        )

        with st.container(
            height=930,
            border=True
        ):

            for group_name, group_items in grouped.items():

                if status_filter == "Extracted fields":
                    group_items = [f for f in group_items if f.get("value")]
                elif status_filter != "All fields":

                    group_items = [
                        field
                        for field in group_items
                        if field["status"] == status_filter
                    ]

                if not group_items:

                    continue

                group_found = sum(
                    field["status"] == "Found"
                    for field in group_items
                )

                st.markdown(
                    '<div class="group-title">'
                    f'{html.escape(group_name)}'
                    f' ({group_found}/{len(group_items)})'
                    '</div>',
                    unsafe_allow_html=True
                )

                for offset in range(
                    0,
                    len(group_items),
                    2
                ):

                    card_columns = st.columns(
                        2,
                        gap="small"
                    )

                    batch = group_items[
                        offset:offset + 2
                    ]

                    for column, field in zip(
                        card_columns,
                        batch
                    ):

                        with column:

                            render_field_card(
                                field,
                                key=(
                                    f'{group_name}_'
                                    f'{offset}_'
                                    f'{field["field"]}'
                                )
                            )


    # ======================================
    # 23. ADDITIONAL RESULT TABS
    # ======================================

    st.divider()

    (
        tab1,
        tab2,
        tab3,
        tab4,
        tab5
    ) = st.tabs([
        "📥 Downloads",
        "🧾 Page-wise text",
        "📊 Detected tables",
        "🔍 Extraction quality",
        "🧠 LayoutLMv3 metadata"
    ])


    # ======================================
    # 24. DOWNLOADS
    # ======================================

    with tab1:

        st.subheader(
            "Download extraction results"
        )

        st.download_button(
            "Download complete extraction ZIP",
            data=st.session_state.output_zip,
            file_name="insurance_verified_extraction.zip",
            mime="application/zip"
        )

        st.download_button(
            "Download fields JSON",
            data=json.dumps(
                fields,
                ensure_ascii=False,
                indent=2
            ),
            file_name="insurance_fields.json",
            mime="application/json"
        )

        st.download_button(
            "Download original PDF",
            data=st.session_state.pdf_bytes,
            file_name=st.session_state.pdf_name,
            mime="application/pdf"
        )

        full_text = "\n\n".join(
            (
                f'===== PAGE {page["page"]} =====\n'
                f'{page["text"]}'
            )
            for page in payload["pages"]
        )

        st.download_button(
            "Download extracted text",
            data=full_text,
            file_name="insurance_full_text.txt",
            mime="text/plain"
        )


    # ======================================
    # 25. PAGE-WISE TEXT
    # ======================================

    with tab2:

        text_page = st.selectbox(
            "Select extracted text page",
            list(
                range(
                    1,
                    payload["processed_pages"] + 1
                )
            ),
            key="text_page"
        )

        page_data = payload["pages"][
            text_page - 1
        ]

        st.caption(
            f'Extraction method: '
            f'{page_data.get("method", "Unknown")}'
        )

        st.code(
            page_data.get("text", ""),
            language=None
        )


    # ======================================
    # 26. DETECTED TABLES
    # ======================================

    with tab3:

        table_page = st.selectbox(
            "Select table page",
            list(
                range(
                    1,
                    payload["processed_pages"] + 1
                )
            ),
            key="table_page"
        )

        page_data = payload["pages"][
            table_page - 1
        ]

        tables = page_data.get(
            "tables",
            []
        )

        st.metric(
            "Detected tables",
            len(tables)
        )

        if not tables:

            st.info(
                "No native PDF tables detected "
                "on this page. Spatial extraction "
                "may still identify fields."
            )

        for table in tables:

            st.markdown(
                f'#### Table {table["table_id"]}'
            )

            records = []

            for row in table.get(
                "rows",
                []
            ):

                record = {}

                for cell in row.get(
                    "cells",
                    []
                ):

                    column_name = (
                        f'Column {cell["column"] + 1}'
                    )

                    record[column_name] = (
                        cell.get("text", "")
                    )

                records.append(record)

            if records:

                st.dataframe(
                    records,
                    use_container_width=True
                )

            with st.expander(
                "Table coordinates"
            ):

                st.json(table)


    # ======================================
    # 27. EXTRACTION QUALITY
    # ======================================

    with tab4:

        st.subheader(
            "PDF extraction quality"
        )

        q1, q2, q3 = st.columns(3)

        q1.metric(
            "Native text pages",
            payload.get("native_pages", 0)
        )

        q2.metric(
            "OCR pages",
            payload.get("ocr_pages", 0)
        )

        q3.metric(
            "Total words",
            payload.get("total_words", 0)
        )

        quality_records = []

        for page in payload["pages"]:

            quality = page.get(
                "quality",
                {}
            )

            quality_records.append({
                "Page": page["page"],

                "Method": page.get(
                    "method",
                    ""
                ),

                "Words": page.get(
                    "word_count",
                    0
                ),

                "Tables": page.get(
                    "table_count",
                    0
                ),

                "OCR confidence": quality.get(
                    "average_ocr_confidence"
                ),

                "Low confidence words": quality.get(
                    "low_confidence_words"
                )
            })

        st.dataframe(
            quality_records,
            use_container_width=True
        )

        st.info(
            "OCR confidence describes OCR recognition "
            "confidence, not verified insurance "
            "field accuracy."
        )


    # ======================================
    # 28. LAYOUTLMV3 METADATA
    # ======================================

    with tab5:

        model_page = st.selectbox(
            "Select LayoutLMv3 page",
            list(
                range(
                    1,
                    payload["processed_pages"] + 1
                )
            ),
            key="model_page"
        )

        page_data = payload["pages"][
            model_page - 1
        ]

        st.json(
            page_data.get(
                "layoutlmv3_inference",
                []
            )
        )

        st.info(
            "These are pretrained encoder "
            "inference summaries, not trained "
            "semantic field predictions."
        )


# ==========================================
# 29. INITIAL SCREEN
# ==========================================

else:

    st.info(
        "Select an insurance category "
        "or Auto-detect, upload a PDF "
        "and click Extract & verify."
    )

    st.markdown(
        "### Supported insurance categories"
    )

    category_columns = st.columns(5)

    for column, category in zip(
        category_columns,
        TYPES
    ):

        with column:

            with st.container(border=True):

                st.markdown(
                    f"**{category}**"
                )

    st.markdown(
        "### Customer details supported"
    )

    st.write(
        "Policyholder Name, Proposer Name, "
        "Insured Person Name, Mobile Number, Phone Number, "
        "Email ID, Date of Birth, Gender, "
        "Age, Customer ID, Residential Address "
        "and Pincode."
    )

    st.caption(
        "Only information supported by "
        "the uploaded document is extracted. "
        "Unavailable or uncertain fields "
        "require review."
    )
