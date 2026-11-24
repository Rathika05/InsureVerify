
"""
Insurance field extraction engine.

Uses:
1. PDF text and word coordinates
2. Spatial label-value matching
3. Table column extraction
4. Common field validation
5. Customer context verification
6. Exact source bounding boxes

No Qwen or Ollama.
LayoutLMv3 runs in the separate pipeline.
"""

import re
from datetime import datetime

from field_validators import (
    validate_field,
    valid_date,
    valid_money,
    valid_policy_number,
    valid_policyholder,
    valid_person_name,
    valid_phone_number,
    valid_address,
    clean_value
)

from spatial_extractor import (
    build_rows,
    prepare_words,
    extract_candidates,
    extract_table_columns,
    find_value_bbox,
    find_label_spans,
    words_text,
    words_box,
    normalize_text
)


# ==========================================
# 1. INSURANCE CATEGORIES
# ==========================================

TYPES = [
    "Vehicle Insurance",
    "Health Insurance",
    "Family Insurance",
    "Life Insurance",
    "Others"
]


# ==========================================
# 2. COMMON INSURANCE FIELDS
# ==========================================

COMMON = {

    "Insurer": [
        "name of insurer",
        "name of insurance company",
        "insurance company",
        "insurer"
    ],

    "Policy Number": [
        "master policy number",
        "master policy no",
        "policy certificate number",
        "insurance policy number",
        "policy reference number",
        "policy number",
        "policy no",
        "policy id",
        "policy ref"
    ],

    "Policyholder": [
        "name of the master policyholder",
        "name of master policyholder",
        "name of policyholder",
        "name of policy holder",
        "policyholder name",
        "policy holder name",
        "name and address of insured",
        "name & address of insured",
        "insured name and address",
        "name and address of policyholder",
        "policyholder",
        "policy holder",
        "name of insured",
        "insured name"
    ],

    "Proposer Name": [
        "name of the proposer",
        "name of proposer",
        "proposer name",
        "proposer's name",
        "proposer"
    ],

    "Insured Person": [
        "name of insured person",
        "insured person name",
        "name of insured member",
        "insured member name",
        "insured person"
    ],

    "Mobile Number": [
        "policyholder mobile number",
        "proposer mobile number",
        "insured mobile number",
        "customer mobile number",
        "registered mobile number",
        "mobile number",
        "mobile no",
        "mobile","Mobile"
    ],

    "Phone Number": [
        "policyholder phone number",
        "insured contact number",
        "customer phone number",
        "proposer contact number",
        "contact number",
        "contact no",
        "phone number",
        "phone no",
        "telephone number",
        "landline number"
    ],

    "Email ID": [
        "policyholder email",
        "proposer email id",
        "proposer email",
        "insured email",
        "customer email",
        "registered email id",
        "email address",
        "email id",
        "e-mail id",
        "e-mail",
        "email"
    ],

    "Date of Birth": [
        "proposer date of birth",
        "insured date of birth",
        "date of birth",
        "birth date",
        "dob"
    ],

    "Gender": [
        "gender",
        "sex"
    ],

    "Age": [
        "age of insured",
        "insured age",
        "proposer age",
        "age"
    ],

    "Customer ID": [
        "customer identification number",
        "customer number",
        "customer code",
        "customer id",
        "client id",
        "client code"
    ],

    "Residential Address": [
        "policyholder address",
        "proposer address",
        "insured address",
        "customer address",
        "residential address",
        "communication address",
        "correspondence address",
        "mailing address",
        "permanent address",
        "name and address of insured",
        "name & address of insured",
        "insured name and address"
    ],

    "Pincode": [
        "customer pincode",
        "insured pincode",
        "postal code",
        "pin code",
        "pincode"
    ],

    "Policy Start": [
        "date of commencement of policy",
        "date of commencement",
        "commencement date",
        "policy start date",
        "policy inception date",
        "inception date",
        "effective date",
        "risk commencement date",
        "period of insurance from",
        "policy period from",
        "policy start",
        "period from",
        "valid from",
        "cover start date"
    ],

    "Policy End": [
        "policy expiry date",
        "policy expiration date",
        "policy end date",
        "date of expiry",
        "expiry date",
        "expiration date",
        "period of insurance to",
        "policy period to",
        "policy end",
        "valid upto",
        "valid up to",
        "cover end date"
    ],

    "Renewal Date": [
        "annual renewal date",
        "next renewal date",
        "renewal date"
    ],

    "Premium": [
        "total premium received towards assured benefit on date of commencement of policy",
        "total premium received",
        "premium amount",
        "annual premium",
        "premium paid",
        "premium"
    ],

    "Gross Premium": [
        "gross premium",
        "basic premium",
        "base premium"
    ],

    "Net Premium": [
        "net premium"
    ],

    "GST": [
        "total gst",
        "gst amount",
        "goods and services tax"
    ],

    "CGST": [
        "central goods and services tax",
        "cgst amount",
        "cgst"
    ],

    "SGST": [
        "state goods and services tax",
        "sgst amount",
        "sgst"
    ],

    "IGST": [
        "integrated goods and services tax",
        "igst amount",
        "igst"
    ],

    "Total Premium": [
        "total premium payable",
        "total payable premium",
        "total premium including tax",
        "total premium",
        "total amount payable",
        "total payable",
        "premium including gst"
    ],

    "Sum Insured / Assured": [
        "total sum assured",
        "total sum insured",
        "basic sum insured",
        "base sum insured",
        "sum insured amount",
        "sum insured",
        "sum assured",
        "coverage amount"
    ]
}


# ==========================================
# 3. CATEGORY SPECIFIC FIELDS
# ==========================================

SPECIFIC = {

    "Vehicle Insurance": {

        "Registration Number": [
            "vehicle registration number",
            "registration number",
            "registration no",
            "vehicle reg no",
            "reg no",
            "registration mark"
        ],

        "Vehicle Make": [
            "vehicle make",
            "make of vehicle",
            "manufacturer",
            "make"
        ],

        "Vehicle Model": [
            "vehicle model",
            "model of vehicle",
            "model","sub model"
        ],

        "Engine Number": [
            "engine number",
            "engine no"
        ],

        "Chassis Number": [
            "chassis number",
            "chassis no"
        ],

        "Manufacturing Year": [
            "year of manufacture",
            "manufacturing year",
            "mfg year",
            "mfg yr"
        ],

        "Seating Capacity": [
            "seating capacity",
            "seating including driver",
            "seating",
            "seats"
        ],

        "Body Type": [
            "body type",
            "type of body"
        ],

        "IDV": [
            "insured declared value",
            "idv of vehicle",
            "total idv",
            "idv"
        ],

        "CC": [
            "cubic capacity",
            "engine capacity",
            "cc"
        ]
    },

    "Health Insurance": {

        "Waiting Period": [
            "initial waiting period",
            "waiting period"
        ],

        "Co-payment": [
            "co-payment",
            "co payment",
            "copayment",
            "co-pay",
            "copay"
        ],

        "Hospital / TPA": [
            "tpa name",
            "name of tpa",
            "third party administrator",
            "hospital name",
            "name of hospital"
        ],

        "Doctor": [
            "treating doctor",
            "consulting doctor",
            "doctor name",
            "name of doctor"
        ]
    },

    "Family Insurance": {

        "Primary Insured": [
            "primary insured",
            "primary policyholder"
        ],

        "Family Member": [
            "family member",
            "insured member",
            "dependent name",
            "name of dependent"
        ],

        "Relationship": [
            "relationship to proposer",
            "relationship with proposer",
            "relationship"
        ],

        "Family Floater Cover": [
            "family floater sum insured",
            "floater sum insured",
            "family floater cover"
        ],

        "Waiting Period": [
            "waiting period"
        ],

        "Co-payment": [
            "co-payment",
            "co payment",
            "copayment",
            "co-pay"
        ]
    },

    "Life Insurance": {

        "Life Assured": [
            "name of life assured",
            "life assured name",
            "life assured"
        ],

        "Nominee": [
            "name of nominee",
            "nominee name",
            "nominee","name"
        ],

        "Death Benefit": [
            "death benefit",
            "sum assured on death"
        ],

        "Premium Frequency": [
            "frequency of premium payment",
            "premium payment frequency",
            "premium payment mode",
            "premium mode"
        ],

        "Number of Members": [
            "number of members covered as on the date of commencement of master policy",
            "number of members covered",
            "number of members",
            "total eligible members"
        ]
    },

    "Others": {}
}


# ==========================================
# 4. PROPERTY INSURANCE FIELDS
# ==========================================

PROPERTY_FIELDS = {

    "Bank / Financial Institution": [
        "name of bank financial institution",
        "name of bank/financial institution",
        "name of bank",
        "financial institution",
        "hypothecated to",
        "mortgagee"
    ],

    "Location of Risk": [
        "location of the risk premises",
        "location of the risk / premises",
        "location of risk",
        "risk location",
        "risk address",
        "insured premises"
    ],

    "Occupation": [
        "occupied as",
        "nature of occupancy",
        "occupancy",
        "occupation"
    ],

    "Terrorism Premium": [
        "terrorism premium",
        "add terrorism",
        "terrorism"
    ]
}


# ==========================================
# 5. KNOWN INSURERS
# ==========================================

INSURERS = [

    "Life Insurance Corporation of India",
    "LIC of India",

    "The New India Assurance Co. Ltd.",
    "The New India Assurance Company Limited",

    "United India Insurance Company Limited",
    "National Insurance Company Limited",
    "The Oriental Insurance Company Limited",

    "Star Health and Allied Insurance Company Limited",
    "HDFC ERGO General Insurance Company Limited",
    "ICICI Lombard General Insurance Company Limited",

    "Bajaj Allianz General Insurance Company Limited",
    "Tata AIG General Insurance Company Limited",
    "SBI General Insurance Company Limited",

    "Care Health Insurance Limited",
    "Niva Bupa Health Insurance Company Limited",
    "Kshema General Insurance Limited",

    "Aditya Birla Health Insurance Company Limited",
    "ManipalCigna Health Insurance Company Limited",
    "Reliance General Insurance Company Limited",

    "Royal Sundaram General Insurance Co. Limited",
    "Royal Sundaram General Insurance Company Limited",

    "Universal Sompo General Insurance Company Limited",
    "Cholamandalam MS General Insurance Company Limited",
    "Future Generali India Insurance Company Limited",
    "Go Digit General Insurance Limited",
    "ACKO General Insurance Limited",

    "HDFC Life Insurance Company Limited",
    "SBI Life Insurance Company Limited",
    "ICICI Prudential Life Insurance Company Limited"
]


# ==========================================
# 6. REGULAR EXPRESSIONS
# ==========================================

DATE_RE = re.compile(
    r"(?<!\d)(?:"
    r"\d{4}[/.-]\d{1,2}[/.-]\d{1,2}|"
    r"\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|"
    r"\d{1,2}-[A-Za-z]{3,9}-\d{4}"
    r")(?!\d)"
)

MONEY_RE = re.compile(
    r"(?<![\w.,])"
    r"(?:₹|Rs\.?|INR)?\s*"
    r"(\d[\d,]*(?:\.\d{1,2})?)"
    r"(?![\w.,])",
    re.I
)

POLICY_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-z0-9]+"
    r"(?:\s*[-/]\s*[A-Za-z0-9]+)*"
    r"(?![A-Za-z0-9])"
)

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,63}",
    re.I
)

PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+91[\s-]*)?"
    r"[6-9](?:[\s-]?\d){9}"
    r"(?!\d)"
)

PINCODE_RE = re.compile(
    r"(?<!\d)[1-9]\d{5}(?!\d)"
)

PROPERTY_RE = re.compile(
    r"\b(?:"
    r"fire and special perils|"
    r"property insurance|"
    r"industrial all risks|"
    r"fire insurance|"
    r"bharat griha raksha|"
    r"bharat sookshma udyam suraksha|"
    r"bharat laghu udyam suraksha"
    r")\b",
    re.I
)


# ==========================================
# 7. CUSTOMER CONTEXT
# ==========================================

CUSTOMER_FIELDS = {

    "Policyholder",
    "Proposer Name",
    "Insured Person",
    "Mobile Number",
    "Phone Number",
    "Email ID",
    "Date of Birth",
    "Gender",
    "Age",
    "Customer ID",
    "Residential Address",
    "Pincode"
}

CONTACT_FIELDS = {

    "Mobile Number",
    "Phone Number",
    "Email ID",
    "Residential Address",
    "Pincode"
}

CUSTOMER_CONTEXT = re.compile(
    r"\b(?:proposer|policy\s*holder|"
    r"insured|customer|applicant|"
    r"policy owner)\b",
    re.I
)

NON_CUSTOMER_CONTEXT = re.compile(
    r"\b(?:agent|broker|intermediary|"
    r"customer care|customer service|"
    r"helpline|toll free|grievance|"
    r"branch office|registered office|"
    r"corporate office|issuing office|"
    r"servicing office|"
    r"hospital|tpa|third party administrator|"
    r"claims assistance|ombudsman)\b",
    re.I
)

STOP_TEXT = re.compile(
    r"\b(?:following documents|"
    r"policy issuing|faq|how do i|"
    r"above changes|"
    r"authorised signatory|"
    r"authorized signatory)\b",
    re.I
)

PERIOD_RE = re.compile(
    r"\b(?:policy period|"
    r"period of insurance|"
    r"period of cover|"
    r"insurance period)\b",
    re.I
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
    "Sum Insured / Assured",
    "Family Floater Cover",
    "Death Benefit",
    "Terrorism Premium",
    "IDV"
}

DATE_FIELDS = {

    "Policy Start",
    "Policy End",
    "Renewal Date",
    "Date of Birth"
}

MULTILINE_FIELDS = {

    "Residential Address",
    "Location of Risk"
}


# ==========================================
# 8. ALL LABELS AND BOUNDARIES
# ==========================================


# Expanded insurance labels. Keep field ownership explicit.
def _extend_schema(schema, entries):
    for field, labels in entries.items():
        schema[field] = list(dict.fromkeys(schema.get(field, []) + labels.split('|')))

_extend_schema(COMMON, {
    "Policy Number": "policy no.|policy reference no|current policy number|current policy no|policy certificate no|policy document number|policy #|policy serial number",
    "Previous Policy Number": "previous policy number|previous policy no|prev policy no|expiring policy number|old policy number|earlier policy no",
    "Cover Note Number": "cover note number|cover note no|covernote no|cover note reference",
    "Cover Note Date": "cover note date|date of cover note|covernote date",
    "Policy Issuance Date": "policy issuance date|policy issue date|date of issue|date of issuance|issued on|policy issued on",
    "UIN": "uin|unique identification number|product uin|product unique identification number",
    "Policyholder": "master policyholder name|name of policy owner|policy owner name|name of policy holder",
    "Insured Name": "insured name|name of insured|name of the insured|named insured|insured's name|insured full name",
    "Proposer Name": "applicant name|name of applicant|name of the proposer|proposer full name",
    "Insured Code": "insured code|insured id|insured customer code|insured identification number|insured person code",
    "Customer ID": "customer ref no|customer identification no|client number|client identification number|customer reference number",
    "Insured GSTIN": "insured gstin|insured gst no|insured gstin no|customer gstin|policyholder gstin|proposer gstin|gstin of insured",
    "Insured PAN": "insured pan|insured pan no|customer pan|proposer pan number|policyholder pan",
    "Mobile Number": "mobile no.|mobile telephone|cell number|cell phone number|insured mobile no|customer mobile no|proposer mobile no|registered mobile|mobile phone",
    "Phone Number": "telephone no|tel no|landline no|insured phone no|customer telephone number|phone|telephone",
    "Email ID": "e mail|e mail id|e mail address|insured email id|customer email id|policyholder email id|proposer email address",
    "Residential Address": "address of insured|address of proposer|address of policyholder|insured correspondence address|contact address|address for communication|address for correspondence|correspondence address",
    "Registration Address": "registration address|vehicle registration address|registered address of vehicle",
    "Pincode": "pin no|postal pin|postal pin code|insured pin code|customer pin code|zip code",
    "Policy Start": "policy commencement date|coverage start date|insurance start date|risk start date|cover commencement date|from date & time|from date and time",
    "Policy End": "policy expiration|coverage end date|insurance expiry date|risk end date|cover expiry date|to date & time|to date and time|valid till|valid until",
    "Renewal Date": "renewal due date|date of renewal|next policy renewal date",
    "Issue Office Name": "issue office name|issuing office name|policy issuing office name|name of issuing office|name of issue office|issuing branch name|policy issuing office",
    "Issue Office Code": "issue office code|issuing office code|policy issuing office code|office code|issuing branch code|branch office code",
    "Issue Office GSTIN": "issue office gstin|issuing office gstin|office gstin|branch gstin|insurer gstin|company gstin|gstin of insurer",
    "Issue Office Address": "issue office address|issuing office address|address of issuing office|issuing branch address|policy issuing office address|office address",
    "Office Email": "office email|office email id|issuing office email|branch email|insurer email|office e mail",
    "Office Phone": "office phone|office telephone|issuing office phone|branch phone number|office contact number",
    "Servicing Office": "policy servicing office|servicing office name|servicing branch",
    "IRDAI Registration Number": "irdai registration number|irda regn no|irdai reg no|irda registration no",
    "SAC Code": "sac code|service accounting code|sac number",
    "Basic Premium": "basic premium|base premium|basic insurance premium|base insurance premium",
    "Gross Premium": "gross insurance premium|gross premium amount|gross premium before tax",
    "Net Premium": "net insurance premium|net premium amount|net premium before tax",
    "GST": "gst payable|total gst amount|goods & services tax amount|gst tax amount",
    "CGST": "central gst amount|central tax amount|cgst payable",
    "SGST": "state gst amount|state tax amount|sgst payable",
    "IGST": "integrated gst amount|integrated tax amount|igst payable",
    "Service Tax": "service tax amount|service tax|service tax including swachh bharat cess",
    "Stamp Duty": "stamp duty|stamp duty amount|stamp duty paid|policy stamp duty",
    "Total Premium": "grand total premium|premium inclusive of tax|total premium inclusive of gst|total premium including service tax|final premium payable|total amount of premium",
    "Receipt Number": "receipt number|receipt no|premium receipt no|payment receipt number",
    "Receipt Date": "receipt date|date of receipt|premium receipt date",
    "Payment Mode": "payment mode|mode of payment|payment method",
    "Fund Transfer Number": "fund transfer no|fund transfer number|transaction reference number|transaction id|utr number",
    "Payment Date": "payment date|date of payment|transaction date",
    "Agent Name": "agent name|name of agent|insurance agent name",
    "Agent Code": "agent code|agent id|insurance agent code",
    "Intermediary Name": "intermediary name|name of intermediary|intermediary full name",
    "Intermediary Code": "intermediary code|intermediary id|intermediary number",
    "Broker Name": "broker name|name of broker|insurance broker name",
    "Broker Code": "broker code|broker id|insurance broker code",
    "Nominee": "nominee name|name of nominee|nominee|nominee owner driver|name of the nominee",
    "Nominee Relationship": "nominee relationship|relationship of nominee|nominee relation|relationship with nominee",
})
# Do not equate distinct roles or premium types.
for field, rejected in {"Proposer Name": {"insured name"}, "Policyholder": {"insured name", "name of insured"}, "Gross Premium": {"basic premium", "base premium"}}.items():
    COMMON[field] = [a for a in COMMON[field] if a.lower() not in rejected]
_extend_schema(SPECIFIC["Vehicle Insurance"], {
    "Registration Number": "registration no.|vehicle registration no|vehicle reg number|regn no|registration mark number|vehicle number",
    "Vehicle Make": "make of motor vehicle|vehicle manufacturer|manufacturer name",
    "Vehicle Model": "model variant|model - variant|model of motor vehicle|vehicle model name|model name",
    "Variant": "vehicle variant|model variant name|variant",
    "Engine Number": "engine no.|engine serial number|motor engine number",
    "Chassis Number": "chassis no.|chassis serial number|vin|vehicle identification number",
    "Manufacturing Year": "mfg. yr|mfg. year|manufacture year|year of mfg|year of manufacturing",
    "Seating Capacity": "seating capacity including driver|seating cap|no of seats|number of seats",
    "CC": "cubic capacity cc|engine cc|cc hp|cubic capacity horsepower",
    "RTO": "rto|rto location|registering authority|registration authority",
    "Net Liability Premium": "net liability premium|net liability amount|net third party liability premium",
    "Total Package Premium": "total package premium|package premium total|package premium",
    "Basic Third Party Liability": "basic third party liability|basic tp premium|basic third party premium",
    "Own Damage Premium": "own damage premium|basic own damage premium|od premium",
    "PA Cover Amount": "pa cover amount|personal accident sum insured|owner driver pa sum insured",
    "PA Cover Premium": "pa cover premium|personal accident premium|owner driver pa premium",
    "No Claim Bonus": "no claim bonus|no claims bonus|ncb|ncb percentage",
    "Previous Insurer": "previous insurer|previous insurance company|name of previous insurer",
    "Geographical Area": "geographical area|geographical limits|geographic area|territorial limits",
    "Vehicle IDV": "vehicle idv|idv of vehicle|vehicle insured declared value",
    "Trailer IDV": "trailer idv|idv of trailer",
    "Electrical Accessories IDV": "electrical accessories idv|idv of electrical accessories",
    "Non-electrical Accessories IDV": "non electrical accessories idv|idv of non electrical accessories",
    "CNG/LPG Kit IDV": "cng lpg kit idv|idv of cng lpg kit",
})
SPECIFIC["Vehicle Insurance"]["IDV"] = ["total idv", "total insured declared value", "insured declared value", "idv"]
for category in ("Health Insurance", "Family Insurance"):
    SPECIFIC[category].pop("Hospital / TPA", None)
    _extend_schema(SPECIFIC[category], {
        "TPA ID": "tpa id|tpa code|tpa identification number|third party administrator code",
        "TPA Name": "tpa name|name of tpa|third party administrator name|third party administrator|name of third party administrator",
        "TPA Address": "tpa address|address of tpa|third party administrator address",
        "TPA Toll Free": "tpa toll free|tpa toll free number|tpa toll free no|tpa helpline number",
        "TPA Phone": "tpa phone|tpa telephone|tpa contact number|tpa phone number",
        "TPA Email": "tpa email|tpa email id|tpa e mail",
        "Hospital Name": "hospital name|name of hospital|network hospital name",
        "Room Rent Limit": "room rent limit|room rent capping|room rent eligibility",
        "Maternity Benefits": "maternity benefits|maternity cover|maternity benefit limit",
        "Number of Members": "number of members covered|number of insured persons|total persons covered|number of employees covered",
        "9 Months Waiting Period Deleted": "9 months waiting period deleted|9 month waiting period deleted",
        "30 Days Waiting Period Deleted": "30 days waiting period deleted|30 day waiting period deleted",
        "2/4 Year Exclusion Deleted": "2/4 year exclusion deleted|2 4 year exclusion deleted",
        "Pre-existing Cover Opted": "pre existing cover opted|pre-existing disease cover opted|pre existing disease cover opted",
    })
_extend_schema(SPECIFIC["Life Insurance"], {
    "Maturity Date": "maturity date|date of maturity|policy maturity date",
    "Premium Frequency": "premium frequency|frequency of payment|mode of premium payment|premium paying frequency",
    "Number of Members": "total members covered|number of lives covered|total insured members",
    "Appointee": "appointee name|name of appointee|appointee",
    "Policy Term": "policy term|term of policy|policy duration",
    "Premium Paying Term": "premium paying term|premium payment term|premium payment duration",
})
_extend_schema(PROPERTY_FIELDS, {
    "Description of Property": "description of property|property description|description of insured property",
    "Building Coverage": "building sum insured|sum insured building|building coverage amount",
    "Machinery Coverage": "machinery sum insured|plant and machinery sum insured|machinery coverage amount",
    "Excess": "policy excess|deductible amount|excess amount|deductible",
    "Special Conditions": "special conditions|special policy conditions",
    "Clauses": "applicable clauses|policy clauses|clauses",
    "Warranties": "applicable warranties|policy warranties|warranties",
})
MONEY_FIELDS.update({"Basic Premium", "Service Tax", "Stamp Duty", "Net Liability Premium", "Total Package Premium", "Basic Third Party Liability", "Own Damage Premium", "PA Cover Amount", "PA Cover Premium", "Vehicle IDV", "Trailer IDV", "Electrical Accessories IDV", "Non-electrical Accessories IDV", "CNG/LPG Kit IDV", "Building Coverage", "Machinery Coverage"})
DATE_FIELDS.update({"Cover Note Date", "Policy Issuance Date", "Receipt Date", "Payment Date", "Maturity Date"})
MULTILINE_FIELDS.update({"Registration Address", "Issue Office Address", "TPA Address", "Description of Property"})

ALL_ALIASES = set()

for aliases in COMMON.values():
    ALL_ALIASES.update(aliases)

for category in SPECIFIC.values():

    for aliases in category.values():
        ALL_ALIASES.update(aliases)

for aliases in PROPERTY_FIELDS.values():
    ALL_ALIASES.update(aliases)

# Additional labels used as boundaries.
# They are not automatically customer fields.

ALL_ALIASES.update([

    "intermediary code",
    "intermediary name",
    "agent code",
    "agent name",
    "broker code",
    "broker name",

    "policy servicing office",
    "policy issuing office",
    "registered office",
    "corporate office",

    "period of insurance",
    "policy period",

    "description of property",
    "special conditions",
    "excess",
    "clauses",
    "warranties",

    "receipt number",
    "receipt details",
    "gstin no",
    "pan no",
    "uin",
    "irda regn no",

    "total"
])

ALL_ALIASES = sorted(
    ALL_ALIASES,
    key=len,
    reverse=True
)


# ==========================================
# 9. INSURANCE TYPE DETECTION
# ==========================================

def guess_type(text):

    text = str(text or "")[:25000].lower()

    if PROPERTY_RE.search(text[:5000]):
        return "Others"

    keywords = {

        "Vehicle Insurance": [
            "motor insurance",
            "vehicle insurance",
            "private car",
            "motor policy",
            "two-wheeler",
            "chassis no",
            "engine no",
            "insured declared value"
        ],

        "Health Insurance": [
            "health insurance",
            "mediclaim",
            "hospitalization",
            "hospitalisation",
            "medical insurance"
        ],

        "Family Insurance": [
            "family floater",
            "family health optima",
            "family health insurance",
            "family insurance",
            "flexi floater"
        ],

        "Life Insurance": [
            "life insurance",
            "group term assurance",
            "group term insurance",
            "death benefit",
            "life assured"
        ]
    }

    scores = {

        category: sum(
            1 for keyword in values
            if keyword in text
        )

        for category, values in keywords.items()
    }

    if scores["Family Insurance"] > 0:

        return "Family Insurance"

    best = max(scores, key=scores.get)

    return best if scores[best] else "Others"


# ==========================================
# 10. COMMON HELPERS
# ==========================================

def _clean(value):

    return clean_value(value)


def _normalize(value):

    return normalize_text(value)


def _compact(value):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value or "").lower()
    )


def _valid_date(value):

    return valid_date(value)


def _valid_name(value):

    return valid_policyholder(value)


def _find_box(words, value):

    return find_value_bbox(
        words,
        value
    )


def _line_groups(words):

    return build_rows({
        "words": words
    })


def missing(field):

    return {

        "field": field,
        "value": "",
        "page": None,
        "bbox": None,
        "status": "Not found",
        "evidence": ""
    }


# ==========================================
# 11. SCHEMA CREATION
# ==========================================

def get_schema(payload, kind):

    schema = {

        **COMMON,
        **SPECIFIC.get(kind, {})
    }

    pages = payload.get("pages") or []

    text = "\n".join(
        str(page.get("text", ""))
        for page in pages
    )

    if kind == "Others" and PROPERTY_RE.search(text):

        schema.update(PROPERTY_FIELDS)

    return schema


# ==========================================
# 12. CUSTOMER OWNERSHIP VERIFICATION
# ==========================================

def _customer_context(
    field,
    alias,
    rows,
    index
):

    if field not in CONTACT_FIELDS:

        return True

    current = rows[index]["text"]

    if NON_CUSTOMER_CONTEXT.search(current):

        return False

    if CUSTOMER_CONTEXT.search(alias):

        return True

    if CUSTOMER_CONTEXT.search(current):

        return True

    for previous in reversed(
        rows[max(0, index - 5):index]
    ):

        text = previous["text"]

        if NON_CUSTOMER_CONTEXT.search(text):

            return False

        if CUSTOMER_CONTEXT.search(text):

            return True

    return False


def candidate_customer_context(
    page,
    field,
    candidate
):

    if field not in CONTACT_FIELDS:

        return True

    rows = build_rows(page)

    box = candidate.get("bbox")

    if not box or not rows:

        return False

    if NON_CUSTOMER_CONTEXT.search(
        candidate.get("evidence", "")
    ):

        return False

    alias = candidate.get("label", "")

    if CUSTOMER_CONTEXT.search(alias):

        return True

    value_y = (box[1] + box[3]) / 2

    index = min(

        range(len(rows)),

        key=lambda i: abs(
            rows[i]["y"] - value_y
        )
    )

    return _customer_context(
        field,
        alias,
        rows,
        index
    )


# ==========================================
# 13. FIELD VALUE CLEANING
# ==========================================

def extract_money(raw):

    raw = _clean(raw)

    matches = []

    for match in MONEY_RE.finditer(raw):

        # Do not treat GST percentage as money.
        following = raw[match.end():]

        if re.match(r"\s*%", following):

            continue

        value = match.group(1)

        if valid_money(value):

            matches.append(value)

    # Multiple monetary values are ambiguous.
    if len(matches) == 1:

        return matches[0]

    return ""


def extract_policy_number(raw):

    raw = _clean(raw)

    if valid_policy_number(raw):

        return raw

    values = []

    for match in POLICY_RE.finditer(raw):

        value = match.group()

        if valid_policy_number(value):

            values.append(value)

    if len(values) == 1:

        return values[0]

    return ""


_original_validate_field = validate_field

def validate_field(field, value):
    value = clean_value(value)
    if field in MONEY_FIELDS:
        return valid_money(value)
    if field in DATE_FIELDS:
        return valid_date(value)
    if field in {"Insured GSTIN", "Issue Office GSTIN"}:
        return bool(re.fullmatch(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]", value.upper()))
    if field == "Insured PAN":
        return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", value.upper()))
    if field in {"Office Email", "TPA Email"}:
        return bool(EMAIL_RE.fullmatch(value))
    if field in {"Office Phone", "TPA Phone", "TPA Toll Free"}:
        return bool(re.fullmatch(r"[+0-9() -]{7,24}", value)) and 7 <= len(re.sub(r"\D", "", value)) <= 15
    if field in {"Insured Name", "Nominee"}:
        return valid_policyholder(value)
    if field in {"Issue Office Address", "TPA Address", "Registration Address"}:
        return valid_address(value)
    if field == "Previous Policy Number":
        return valid_policy_number(value)
    return _original_validate_field(field, value)


def _value_from(raw, field):

    raw = _clean(raw)

    if not raw:

        return ""

    if STOP_TEXT.search(raw):

        return ""

    if field in {"Policy Number", "Previous Policy Number"}:

        return extract_policy_number(raw)

    if field in DATE_FIELDS:

        if valid_date(raw):

            return raw

        dates = [

            match.group()

            for match in DATE_RE.finditer(raw)

            if valid_date(match.group())
        ]

        return dates[0] if len(dates) == 1 else ""

    if field in MONEY_FIELDS:

        return extract_money(raw)

    if field == "Email ID":

        emails = EMAIL_RE.findall(raw)

        if len(emails) == 1:

            value = emails[0]

            if validate_field(field, value):

                return value

        return ""

    if field == "Mobile Number":

        if validate_field(field, raw):

            return raw

        phones = PHONE_RE.findall(raw)

        if len(phones) == 1:

            if validate_field(field, phones[0]):

                return phones[0]

        return ""

    if field == "Phone Number":

        return raw if valid_phone_number(raw) else ""

    if field == "Pincode":

        matches = PINCODE_RE.findall(raw)

        return matches[0] if len(matches) == 1 else ""

    if field == "Policyholder":

        return raw if valid_policyholder(raw) else ""

    if field in MULTILINE_FIELDS:

        if field == "Residential Address" and NON_CUSTOMER_CONTEXT.search(raw):

            return ""

        return raw if valid_address(raw) else ""

    if validate_field(field, raw):

        return raw

    return ""


# ==========================================
# 14. CANDIDATE CONSTRUCTION
# ==========================================

def make_result_candidate(
    page,
    field,
    value,
    words,
    evidence,
    score=100,
    verified=True
):

    value = _clean(value)

    if not validate_field(field, value):

        return None

    box = find_value_bbox(
        words,
        value
    )

    if not box:

        return None

    return {

        "value": value,
        "page": page.get("page"),
        "bbox": box,
        "evidence": str(evidence)[:500],
        "score": score,
        "verified": bool(verified)
    }


# ==========================================
# 15. SPATIAL FIELD EXTRACTION
# ==========================================

def _candidates(page, field, aliases):

    candidates = extract_candidates(

        page=page,

        aliases=aliases,

        all_aliases=ALL_ALIASES,

        multiline=field in MULTILINE_FIELDS
    )

    for candidate in candidates:

        label = candidate.get("label", "")

        raw = candidate.get("value", "")

        method = candidate.get("method", "")
        # A short alias must not match inside a different, longer label.
        candidate_box = candidate.get("bbox")
        if candidate_box:
            y = (candidate_box[1] + candidate_box[3]) / 2
            rows = build_rows(page)
            nearby = sorted(rows, key=lambda r: abs(r["y"] - y))[:2]
            own = {_normalize(a) for a in aliases}
            blocked = False
            for row in nearby:
                for span in find_label_spans(row["words"], ALL_ALIASES):
                    full = _normalize(span["label"])
                    short = _normalize(label)
                    if short and full not in own and full != short and re.search(r"(?<!\w)" + re.escape(short) + r"(?!\w)", full):
                        if span["bbox"][2] <= candidate_box[0] + 8 and abs(row["y"] - y) <= 15:
                            blocked = True
            if blocked:
                continue

        if field in CONTACT_FIELDS:

            if not candidate_customer_context(
                page,
                field,
                candidate
            ):

                continue

        # Combined name/address fields need
        # separate handling.

        combined_name_address = (

            "name" in _normalize(label)

            and "address" in _normalize(label)
        )

        if combined_name_address:

            if field == "Residential Address":

                # Handled by dedicated extraction.
                continue

            if field in {"Policyholder", "Insured Name"}:

                if method == "multiline":

                    continue

        value = _value_from(
            raw,
            field
        )

        if not value:

            continue

        score = 100

        if method == "right_cell":

            score = 115

        elif method == "same_row":

            score = 110

        elif method == "multiline":

            score = 120

        elif method == "below_label":

            score = 65

        # Explicit total coverage is preferred.
        if field == "Sum Insured / Assured":

            if "total sum" in _normalize(label):

                score += 20

        if field == "Policy Number":

            if "master policy" in _normalize(label):

                score += 15

        result = make_result_candidate(

            page=page,

            field=field,

            value=value,

            words=candidate.get("words", []),

            evidence=candidate.get("evidence", ""),

            score=score,

            verified=method != "below_label"
        )

        if result:

            yield result


# ==========================================
# 16. COMBINED NAME AND ADDRESS
# ==========================================

def combined_customer_candidates(page, field):

    if field not in {

        "Policyholder",
        "Insured Name",
        "Residential Address",
        "Pincode"
    }:

        return

    aliases = [

        "name and address of insured",
        "name & address of insured",
        "insured name and address",
        "name and address of policyholder"
    ]

    if field == "Insured Name":
        aliases = [a for a in aliases if "policyholder" not in a]
    elif field == "Policyholder":
        aliases = [a for a in aliases if "policyholder" in a]

    candidates = extract_candidates(

        page=page,

        aliases=aliases,

        all_aliases=ALL_ALIASES,

        multiline=True
    )

    for candidate in candidates:

        method = candidate.get("method")

        words = candidate.get("words", [])

        if not words:

            continue

        # First row usually contains the name.
        # Remaining rows may contain the address.

        local_rows = build_rows({

            "words": words
        })

        if not local_rows:

            continue

        if field in {"Policyholder", "Insured Name"}:

            name_words = local_rows[0]["words"]

            name = _value_from(words_text(name_words), field)

            if not valid_policyholder(name):

                continue

            result = make_result_candidate(

                page,
                field,
                name,
                name_words,
                candidate["evidence"],
                score=130,
                verified=method != "below_label"
            )

        elif field == "Residential Address":

            if len(local_rows) < 2:

                continue

            address_words = [

                word

                for row in local_rows[1:]

                for word in row["words"]
            ]

            address = words_text(address_words)

            if not valid_address(address):

                continue

            result = make_result_candidate(

                page,
                field,
                address,
                address_words,
                candidate["evidence"],
                score=115,

                # The name/address split is inferred.
                verified=False
            )

        else:

            raw = words_text(words)

            pincodes = PINCODE_RE.findall(raw)

            if len(pincodes) != 1:

                continue

            result = make_result_candidate(

                page,
                field,
                pincodes[0],
                words,
                candidate["evidence"],
                score=115,
                verified=method != "below_label"
            )

        if result:

            yield result


# ==========================================
# 17. POLICY PERIOD EXTRACTION
# ==========================================

def _period_candidates(page, field):

    if field not in {

        "Policy Start",
        "Policy End"
    }:

        return

    rows = build_rows(page)

    for index, row in enumerate(rows):

        if not PERIOD_RE.search(row["text"]):

            continue

        nearby = rows[index:index + 3]

        context = " ".join(

            item["text"]

            for item in nearby
        )

        words = [

            word

            for item in nearby

            for word in item["words"]
        ]

        # Explicit FROM and TO labels.
        match = re.search(

            r"\bfrom\b(.*?)\bto\b(.*)",

            context,

            re.I | re.S
        )

        if not match:

            continue

        start_dates = [

            item.group()

            for item in DATE_RE.finditer(
                match.group(1)
            )

            if valid_date(item.group())
        ]

        end_dates = [

            item.group()

            for item in DATE_RE.finditer(
                match.group(2)
            )

            if valid_date(item.group())
        ]

        if len(start_dates) != 1:

            continue

        if len(end_dates) != 1:

            continue

        value = (

            start_dates[0]

            if field == "Policy Start"

            else end_dates[0]
        )

        result = make_result_candidate(

            page,
            field,
            value,
            words,
            context,
            score=140,
            verified=True
        )

        if result:

            yield result


# ==========================================
# 18. PREMIUM TABLE EXTRACTION
# ==========================================

def premium_table_candidates(page, field):

    if field != "Total Premium":

        return

    rows = build_rows(page)

    premium_section = False

    premium_rows = 0

    for row in rows:

        text = row["text"]

        if re.search(

            r"\b(?:annual premium|"
            r"gross premium|net premium|"
            r"premium calculation|"
            r"premium details)\b",

            text,

            re.I
        ):

            premium_section = True

            premium_rows = 0

        if not premium_section:

            continue

        premium_rows += 1

        if premium_rows > 12:

            premium_section = False

            continue

        if re.search(

            r"\b(?:excess|special conditions|"
            r"sum insured|description of property|"
            r"receipt details)\b",

            text,

            re.I
        ):

            premium_section = False

            continue

        match = re.match(

            r"^\s*(?:grand total|"
            r"total premium|"
            r"total payable|total)\b",

            text,

            re.I
        )

        if not match:

            continue

        raw = text[match.end():]

        value = extract_money(raw)

        if not value:

            continue

        result = make_result_candidate(

            page,
            field,
            value,
            row["words"],
            text,
            score=135,
            verified=True
        )

        if result:

            yield result


# ==========================================
# 19. SUM INSURED TABLE EXTRACTION
# ==========================================

def coverage_table_candidates(page, field):

    if field != "Sum Insured / Assured":

        return

    rows = build_rows(page)

    coverage_section = False

    row_count = 0

    for row in rows:

        text = row["text"]

        if re.search(

            r"\b(?:sum insured|sum assured)\b",

            text,

            re.I

        ) and re.search(

            r"\b(?:description|property|"
            r"coverage|benefit)\b",

            text,

            re.I
        ):

            coverage_section = True

            row_count = 0

        if not coverage_section:

            continue

        row_count += 1

        if row_count > 15:

            coverage_section = False

            continue

        if re.search(

            r"\b(?:annual premium|"
            r"gross premium|net premium)\b",

            text,

            re.I
        ):

            coverage_section = False

            continue

        match = re.match(

            r"^\s*(?:total sum insured|"
            r"total sum assured|total)\b",

            text,

            re.I
        )

        if not match:

            continue

        raw = text[match.end():]

        value = extract_money(raw)

        if not value:

            continue

        result = make_result_candidate(

            page,
            field,
            value,
            row["words"],
            text,
            score=140,
            verified=True
        )

        if result:

            yield result


# ==========================================
# 20. VEHICLE TABLE EXTRACTION
# ==========================================

def vehicle_table_candidates(page):

    column_aliases = {

        "Registration Number": [

            "registration no",
            "registration number",
            "reg no"
        ],

        "Vehicle Make": [

            "make",
            "vehicle make"
        ],

        "Vehicle Model": [

            "model",
            "vehicle model"
        ],

        "Engine Number": [

            "engine no",
            "engine number"
        ],

        "Chassis Number": [

            "chassis no",
            "chassis number"
        ],

        "Manufacturing Year": [

            "mfg year",
            "mfg yr",
            "year of manufacture"
        ],

        "Seating Capacity": [

            "seating capacity",
            "seating",
            "seats"
        ],

        "Body Type": [

            "body type",
            "body"
        ],

        "CC": [

            "cc",
            "cubic capacity"
        ]
    }

    tables = extract_table_columns(

        page,

        column_aliases,

        max_data_rows=2
    )

    for table in tables:

        # Avoid treating arbitrary two-column
        # text as a vehicle table.

        if len(table) < 3:

            continue

        for field, candidate in table.items():

            value = _value_from(

                candidate["value"],

                field
            )

            if not value:

                continue

            result = make_result_candidate(

                page,

                field,

                value,

                candidate["words"],

                candidate["evidence"],

                score=105,

                # Inferred column boundaries
                # require review.

                verified=False
            )

            if result:

                yield field, result


# ==========================================
# 21. INSURER EXTRACTION
# ==========================================

def insurer_candidates(page):

    words = prepare_words(page)

    if not words:

        return

    # Known insurer names near the page header.

    header_words = [

        word

        for word in words

        if word["bbox"][1] <= 250
    ]

    for insurer in INSURERS:

        box = find_value_bbox(

            header_words,

            insurer
        )

        if not box:

            continue

        yield {

            "value": insurer,

            "page": page.get("page"),

            "bbox": box,

            "evidence": insurer,

            "score": 140,

            "verified": True
        }

    # Support insurer names not present
    # in the known insurer list.

    for row in build_rows(page):

        if row["y"] > 180:

            break

        text = row["text"]

        if len(text.split()) > 20:

            continue

        if not re.search(

            r"\b(?:insurance|assurance)\b",

            text,

            re.I
        ):

            continue

        if not re.search(

            r"\b(?:company|corporation|"
            r"limited|ltd|co)\b",

            text,

            re.I
        ):

            continue

        if re.search(

            r"\b(?:policy schedule|"
            r"policy number|"
            r"customer care)\b",

            text,

            re.I
        ):

            continue

        result = make_result_candidate(

            page,
            "Insurer",
            text,
            row["words"],
            text,
            score=100,
            verified=False
        )

        if result:

            yield result


# ==========================================
# 22. CANDIDATE RANKING
# ==========================================

def choose_best(field, candidates):

    if not candidates:

        return missing(field)

    unique = {}

    for candidate in candidates:

        key = (

            _compact(candidate["value"]),

            candidate["page"],

            tuple(candidate["bbox"])
        )

        old = unique.get(key)

        if old is None or (

            candidate["verified"],

            candidate["score"]

        ) > (

            old["verified"],

            old["score"]
        ):

            unique[key] = candidate

    candidates = list(unique.values())

    candidates.sort(

        key=lambda item: (

            not item["verified"],

            -item["score"],

            item["page"] or 9999
        )
    )

    best = candidates[0]

    # Compare competing candidates with
    # similar evidence strength.

    competing = {

        _compact(item["value"])

        for item in candidates

        if item["verified"]

        and item["score"] >= best["score"] - 10
    }

    conflict = len(competing) > 1

    status = (

        "Found"

        if best["verified"] and not conflict

        else "Needs review"
    )

    return {

        "field": field,

        "value": best["value"],

        "page": best["page"],

        "bbox": best["bbox"],

        "status": status,

        "evidence": best["evidence"]
    }


# ==========================================
# 23. FINAL FIELD EXTRACTION
# ==========================================

def extract_fields(payload, kind):

    schema = get_schema(

        payload,

        kind
    )

    pages = payload.get("pages") or []

    collected = {

        field: []

        for field in schema
    }

    for page in pages:

        for field, aliases in schema.items():

            collected[field].extend(

                _candidates(

                    page,

                    field,

                    aliases
                )
            )

            if field in {

                "Policyholder",
                "Insured Name",
                "Residential Address",
                "Pincode"
            }:

                collected[field].extend(

                    combined_customer_candidates(

                        page,

                        field
                    )
                )

            if field in {

                "Policy Start",
                "Policy End"
            }:

                collected[field].extend(

                    _period_candidates(

                        page,

                        field
                    )
                )

            if field == "Total Premium":

                collected[field].extend(

                    premium_table_candidates(

                        page,

                        field
                    )
                )

            if field == "Sum Insured / Assured":

                collected[field].extend(

                    coverage_table_candidates(

                        page,

                        field
                    )
                )

            if field == "Insurer":

                collected[field].extend(

                    insurer_candidates(page)
                )

        if kind == "Vehicle Insurance":

            for field, candidate in vehicle_table_candidates(page):

                if field in collected:

                    collected[field].append(candidate)

    result = []

    for field in schema:

        result.append(

            choose_best(

                field,

                collected[field]
            )
        )

    by_field = {item["field"]: item for item in result}
    pin = by_field.get("Pincode")
    address = by_field.get("Residential Address")
    if pin and not pin["value"] and address and address["value"] and address.get("bbox"):
        codes = PINCODE_RE.findall(address["value"])
        if len(set(codes)) == 1:
            for page in pages:
                if page.get("page") != address["page"]:
                    continue
                b = address["bbox"]
                words = [w for w in prepare_words(page) if w["bbox"][0] >= b[0]-2 and w["bbox"][1] >= b[1]-2 and w["bbox"][2] <= b[2]+2 and w["bbox"][3] <= b[3]+2]
                box = find_value_bbox(words, codes[0])
                if box:
                    pin.update(value=codes[0], page=address["page"], bbox=box, status=address["status"], evidence=address["evidence"])
    return result

# Dataset-derived label extensions (native-text pages).
_extend_schema(COMMON, {
 'Policy Name':'product|product name|insurance product|name of the scheme',
 'Policy Issuance Date':'policy issued on|document date',
 'Cover Note Date':'cover note dt|covernote dt',
 'Insured Code':"insured's code|insureds code",
 'Insured Name':"insured's name|name of the insured",
 'Insured GSTIN':'gstin of customer|gstin of insured|customer gst number',
 'Issue Office GSTIN':'company gst no|company gst number|fgi gstin number',
 'Company PAN':'company pan|nia pan number|insurer pan number',
 'Invoice Number':'invoice no|tax invoice no|tax invoice number',
 'Product Code':'product code',
 'Collection Number':'collection no|collection number',
 'Collection Date':'collection date|collection dt',
 'Development Officer Code':'dev.off.code|dev off code|development officer code',
 'Lead Number':'lead/breakin no|lead breakin no',
 'Agency Code':'agency code|agency/broker code',
 'Agency Name':'agency name',
 'EMF Code':'emf code',
 'Fulfiller Code':'fulfiller code',
 'Business Source Code':'business source code',
 'Debit Note Number':'debitnote number|debit note no',
 'Place of Supply':'place of supply|place of supply state code',
 'Insured State':'insured state & code|insured state',
 'SAC Code':'services accounting code|accounting code of service',
 'Total Premium':'final premium|total premium for the policy period|total premium rounded off',
 'Gross Premium':'gross premium paid',
 'CGST':'central gst|add cgst', 'SGST':'state gst|add sgst',
 'Special Discount':'special discount', 'Additional Discount':'additional discount',
 'Additional Loading':'additional loading', 'Terrorism Surcharge':'terrorism surcharge',
 'Highest Sum Insured':'highest sum insured',
 'Scope of Cover':'scope of cover', 'Risk Covered':'risk covered',
 'Special Exclusions':'special exclusions', 'Special Perils':'special perils',
 'Corporate Identification Number':'corporate identification number|cin',
 'Channel Name':'channel name', 'Bank RM Employee Code':'bank rm employee code',
 'Policy Printed Date':'policy printed on', 'Policy Printed By':'policy printed by',
})
_extend_schema(SPECIFIC['Vehicle Insurance'], {
 'Registration Number':'vehicle no|vehicle no.',
 'Vehicle Model':'make/model of vehicle|make and model of vehicle insured',
 'RTO':'rto where vehicle is/will be registered',
 'Vehicle Class':'class of vehicle', 'Vehicle Usage':'usage of the vehicle',
 'Vehicle Age':'age of vehicle', 'Area Extension':'area extension',
 'Hypothecation':'hypothecation agreement with|hypothecated to',
 'Own Damage Premium':'basic od premium|basic premium on vehicle|basic premium on vehicle and accessories',
 'Net Own Damage Premium':'net own damage premium (a)|total own damage premium (a) (rounded off)',
 'Net Liability Premium':'net liability premium (b)|total liability premium (b)',
 'Total Package Premium':'total premium (a+b)|total annual premium (a+b)',
 'Add-on Premium':'add-on premium|add on premium',
 'Compulsory Deductible':'compulsory deductible|compulsory excess',
 'Voluntary Deductible':'voluntary deductible', 'Additional Excess':'additional excess',
 'Legal Liability Driver Premium':'legal liability (wc) to driver (imt-28)|legal liability to paid driver',
 'Previous Policy End':'expiring policy expiry date|previous policy expiry date',
 'Previous No Claim Bonus':'no claim bonus % under expiring policy',
 'Previous Claim':'is there any claim in expiring policy',
 'Renewal IDV':'vehicle idv on renewal',
 'Nominee Relationship':'nominee relationship with insured',
 'Nominee Age':'nominee age in y or m|nominee age', 'Nominee Share':'nominee %',
 'Appointee Relationship':'relationship of appointee with nominee',
 'Motor Liability Period':'motor liability period',
 'Registration Address':'registration address of the insured',
 'Residential Address':'communication address of the insured',
})
for _category in ('Health Insurance','Family Insurance'):
    _extend_schema(SPECIFIC[_category], {'Pre-existing Disease':'pre existing disease|pre-existing disease','Number of Members':'number of insured persons|total members covered','Member ID':'member id|member code|insured member id','Normal Delivery Limit':'normal delivery limit|normal delivery','Caesarean Limit':'caesarean limit|caesarean delivery','Zone Opted':'zone opted','Ambulance Limit':'ambulance charges|ambulance limit'})
_extend_schema(SPECIFIC['Others'], {'Number of Members':'number of members covered|total members covered','Accident Cover':'personal accident cover|accident cover'})
MONEY_FIELDS.update({'Highest Sum Insured','Special Discount','Terrorism Surcharge','Net Own Damage Premium','Add-on Premium','Compulsory Deductible','Voluntary Deductible','Additional Excess','Legal Liability Driver Premium','Renewal IDV','Normal Delivery Limit','Caesarean Limit','Ambulance Limit'})
DATE_FIELDS.update({'Collection Date','Previous Policy End','Policy Printed Date'})
ALL_ALIASES = sorted(set(ALL_ALIASES).union(*(set(v) for s in [COMMON, PROPERTY_FIELDS,*SPECIFIC.values()] for v in s.values())),key=len,reverse=True)


# Expanded non-motor schedules, based on dataset documents 001–011, 050–054.
_extend_schema(COMMON, {
 'Proposal Number':'proposal no|proposal number|proposal reference number',
 'Proposal Date':'date of proposal|proposal date|date of application',
 'Proposer Code':'proposer code|proposer id', 'Customer Name':'customer name|name of customer',
 'CKYC Number':'cust ckyc no|ckyc number|ckyc no|customer ckyc number',
 'Policy Category':'policy category|category of policy', 'Policy Term':'policy term|term of policy|policy duration',
 'First Policy Inception Date':'date of inception of first policy|first policy inception date',
 'Client Type':'client type|customer type', 'Co-insurance Share':'own share|co insurance share',
 'Financier Details':'financier(s) details|financier details|financiers details',
 'UIN':'unique identification no|unique id|uin number', 'SAC Code':'sac',
 'Previous Policy Number':'prev. policy no.|prev. policy|previous policy no.',
 'Tax Invoice Number':'gst invoice no|tax invoice no|tax invoice number',
 'Policy Issuance Date':'date of issue|policy issued on',
 'Entered By':'entered by', 'Approved By':'approved by', 'Examined By':'examined by',
})
health = {
 'Number of Employees':'no. of employees / members covered|number of employees covered|no of employees covered',
 'Number of Persons Covered':'no. of persons covered|number of persons covered|total persons covered',
 'Maternity Benefits':'maternity benefits opted|maternity benefits|maternity cover',
 'Normal Delivery Limit':'normal delivery limit|normal delivery',
 'Caesarean Limit':'caesarian section limit|caesarean section limit|caesarian section|caesarean delivery limit',
 '9 Months Waiting Period Deleted':'deletion of 9 months waiting period|9 months waiting period deleted',
 '30 Days Waiting Period Deleted':'deletion of 30 days waiting period|30 days waiting period deleted',
 '2/4 Year Exclusion Deleted':'deletion of 2/4 year exclusion|deletion of 2 4 year exclusion',
 'Pre-existing Cover Opted':'pre-existing cover opted|pre existing cover opted',
 'Ambulance Limit':'limit of additional ambulance charges per person|limit of additional ambulance charges|ambulance limit',
 'Additional Cover Opted':'additional cover opted|additional coverage opted',
 'Corporate Buffer':'corporate buffer|corporate buffer limit',
 'Installment Facility':'installment facility optn|installment facility option|instalment facility option',
 'Premium Frequency':'premium payment frequency|frequency of premium payment',
 'Installment Amount':'installment amount rs|installment amount|instalment amount',
 'Bonus':'bonus|cumulative bonus|no claim bonus amount',
 'Limit of Coverage':'limit of coverage|coverage limit', 'Recharge Benefit':'recharge benefit|recharge amount',
 'Family Floater Cover':'basic floater sum insured|family floater sum insured|floater sum insured',
 'Family Size':'scheme description (family size)|scheme description family size|family size',
 'Voluntary Co-payment':'voluntary co-payment|voluntary co payment',
 'TPA Fax':'tpa fax|tpa fax no', 'Overall Policy Limit':'overall policy limit',
 'Limit per Family':'limit per family|family limit', 'Cover Name':'name of cover|cover name',
}
for category in ('Health Insurance','Family Insurance'):
    _extend_schema(SPECIFIC[category], health)
    _extend_schema(SPECIFIC[category], {'Insured Person':'name of the insured person|name of insured member|name of the insured|insured member name','Relationship':'relationship to proposer|relationship with proposer|relationship','Member ID':'member id|insured person id|member code'})
_extend_schema(SPECIFIC['Life Insurance'], {
 'Master Policy Number':'master policy number|master policy no', 'Scheme Name':'name of the scheme|scheme name',
 'Group Type':'type of group|group type', 'Divisional Office':'divisional office', 'PGS Unit Office':'p&gs unit office|pgs unit office',
 'Minimum Entry Age':'minimum entry age|min entry age', 'Maximum Entry Age':'maximum entry age|max entry age',
 'Maximum Cover Ceasing Age':'maximum cover ceasing age|cover ceasing age',
 'Premium Due Date':'premium due date|next premium due date', 'Premium Frequency':'frequency of premium payment',
 'Number of Members':'number of members covered as on the date of commencement of master policy|number of members covered|total eligible members',
 'Rider Opted':'rider opted|details of rider opted', 'Rider Sum Assured':'rider sum assured',
 'Death Benefit':'death benefit|sum assured on death',
})
_extend_schema(SPECIFIC['Others'], {
 'Number of Students':'no of students|no. of students|number of students',
 'Number of Parents':'no of parents|number of parents', 'Number of Lives':'no.of lives|no of lives|number of lives',
 'Medical Expenses per Student':'medical expenses per student (inclusive of opd)|medical expenses per student',
 'Limit per Student':'limit per student', 'Limit per Accident':'limit per accident',
 'Parents Sum Insured':'total si of parents or guardian for payment of tuition and hostel fees',
 'Tuition and Hostel Fees':'payment of tuition and hostel fee for remaining semesters',
 'Risk Group':'risk group|risk class', 'Accidental Death Cover':'accidental death cover|accidental death benefit',
 'Permanent Total Disablement':'permanent total disablement|permanent total disability',
 'Permanent Partial Disablement':'permanent partial disablement|permanent partial disability',
 'Cover Name':'cover name|name of cover', 'Cover Definition':'cover definition',
 'Sum Insured Basis':'sum insured (basis)|sum insured basis',
 'Cremation Charges':'cremation charges', 'Repatriation Benefit':'repatriation of mortal remains|repatriation of remains',
})
_extend_schema(PROPERTY_FIELDS, {'Location of Risk':'location of the risk / premises','Bank / Financial Institution':'name of bank/financial institution','Issued At':'issued at|place of issue','Special Conditions':'special conditions / clauses / warranties applicable if any','Escalation Clause':'escalation clause','Description of Property':'description of property'})
MONEY_FIELDS.update({'Installment Amount','Bonus','Limit of Coverage','Recharge Benefit','Corporate Buffer','Overall Policy Limit','Limit per Family','Rider Sum Assured','Medical Expenses per Student','Limit per Student','Limit per Accident','Parents Sum Insured','Tuition and Hostel Fees','Cremation Charges','Repatriation Benefit'})
DATE_FIELDS.update({'Proposal Date','First Policy Inception Date','Premium Due Date'})
# Death benefit may be a textual scheme rule, not a numeric amount.
MONEY_FIELDS.discard('Death Benefit')
ALL_ALIASES = sorted(set(ALL_ALIASES).union(*(set(v) for s in [COMMON,PROPERTY_FIELDS,*SPECIFIC.values()] for v in s.values())),key=len,reverse=True)
_old_guess_type = guess_type

def guess_type(text):
    t = str(text or '')[:25000].lower()
    if PROPERTY_RE.search(t[:5000]) or re.search(r'group personal accident policy|students safety package insurance',t[:5000]):
        return 'Others'
    if re.search(r'group mediclaim|group health|group medical|flexi floater group',t[:7000]):
        return 'Health Insurance'
    if re.search(r'family health optima|family floater|family health insurance',t[:7000]):
        return 'Family Insurance'
    return _old_guess_type(t)

_old_get_schema = get_schema

def get_schema(payload, kind):
    if kind in ('Auto-detect','Auto','auto',None):
        kind = guess_type('\n'.join(p.get('text','') for p in payload.get('pages',[])[:3]))
    if kind in ('General Insurance','Property Insurance','Fire Insurance','General'):
        kind = 'Others'
    return _old_get_schema(payload,kind)

# Recover multi-column member and rider tables with individual source boxes.
_MEMBER_COLUMNS = {
 'Insured Person':['name of the insured','name of insured person','name of member','insured member name','member name'],
 'Member ID':['member id','member code','insured person id'],
 'Relationship':['relationship','relation with proposer','relationship with proposer'],
 'Age':['age','age in years'], 'Gender':['gender','sex'],
 'Date of Birth':['date of birth','dob'], 'Sum Insured / Assured':['sum insured','sum assured'],
}
_RIDER_COLUMNS = {'Rider Opted':['rider opted','rider name'],'Rider Sum Assured':['rider sum assured'],'UIN':['uin']}
_old_extract_fields = extract_fields

def extract_fields(payload, kind):
    if kind in ('Auto-detect','Auto','auto',None):
        kind = guess_type('\n'.join(p.get('text','') for p in payload.get('pages',[])[:3]))
    if kind in ('General Insurance','Property Insurance','Fire Insurance','General'):
        kind = 'Others'
    results = _old_extract_fields(payload,kind)
    columns = _MEMBER_COLUMNS if kind in ('Health Insurance','Family Insurance') else _RIDER_COLUMNS if kind == 'Life Insurance' else None
    if columns:
        seen = set()
        for page in payload.get('pages',[]):
            for table in extract_table_columns(page,columns,max_data_rows=30):
                required = 'Insured Person' if columns is _MEMBER_COLUMNS else 'Rider Opted'
                if required not in table or len(table)<3:
                    continue
                row = {}
                for field,candidate in table.items():
                    value = _value_from(candidate['value'],field)
                    if not value: continue
                    item = make_result_candidate(page,field,value,candidate['words'],candidate['evidence'],score=100,verified=False)
                    if item: row[field]=item
                if required not in row or len(row)<2: continue
                key = (page.get('page'),tuple(row[required]['bbox']))
                if key in seen: continue
                seen.add(key)
                for field,item in row.items():
                    results.append(dict(field=f'{"Member" if columns is _MEMBER_COLUMNS else "Rider"} {len(seen)} — {field}',value=item['value'],page=item['page'],bbox=item['bbox'],status='Needs review',evidence=item['evidence']))
    return results


# AUTHORITATIVE INLINE ALIAS REGISTRY (all insurance categories).
# Edit these dictionaries to add new PDF label variants.
COMMON = {'Insurer': ['name of insurer', 'name of insurance company', 'insurance company', 'insurer'],
 'Policy Number': ['master policy number',
                   'master policy no',
                   'policy certificate number',
                   'insurance policy number',
                   'policy reference number',
                   'policy number',
                   'policy no',
                   'policy id',
                   'policy ref',
                   'policy no.',
                   'policy reference no',
                   'current policy number',
                   'current policy no',
                   'policy certificate no',
                   'policy document number',
                   'policy #',
                   'policy serial number',
                   'master policy no.',
                   'policy certificate no.'],
 'Policyholder': ['name of the master policyholder',
                  'name of master policyholder',
                  'name of policyholder',
                  'name of policy holder',
                  'policyholder name',
                  'policy holder name',
                  'name and address of insured',
                  'name & address of insured',
                  'insured name and address',
                  'name and address of policyholder',
                  'policyholder',
                  'policy holder',
                  'master policyholder name',
                  'name of policy owner',
                  'policy owner name'],
 'Proposer Name': ['name of the proposer',
                   'name of proposer',
                   'proposer name',
                   "proposer's name",
                   'proposer',
                   'applicant name',
                   'name of applicant',
                   'proposer full name'],
 'Insured Person': ['name of insured person',
                    'insured person name',
                    'name of insured member',
                    'insured member name',
                    'insured person'],
 'Mobile Number': ['policyholder mobile number',
                   'proposer mobile number',
                   'insured mobile number',
                   'customer mobile number',
                   'registered mobile number',
                   'mobile number',
                   'mobile no',
                   'mobile',
                   'mobile no.',
                   'mobile telephone',
                   'cell number',
                   'cell phone number',
                   'insured mobile no',
                   'customer mobile no',
                   'proposer mobile no',
                   'registered mobile',
                   'mobile phone'],
 'Phone Number': ['policyholder phone number',
                  'insured contact number',
                  'customer phone number',
                  'proposer contact number',
                  'contact number',
                  'contact no',
                  'phone number',
                  'phone no',
                  'telephone number',
                  'landline number',
                  'telephone no',
                  'tel no',
                  'landline no',
                  'insured phone no',
                  'customer telephone number',
                  'phone',
                  'telephone',
                  'contact no.',
                  'phone no.'],
 'Email ID': ['policyholder email',
              'proposer email id',
              'proposer email',
              'insured email',
              'customer email',
              'registered email id',
              'email address',
              'email id',
              'e-mail id',
              'e-mail',
              'email',
              'e mail',
              'e mail id',
              'e mail address',
              'insured email id',
              'customer email id',
              'policyholder email id',
              'proposer email address'],
 'Date of Birth': ['proposer date of birth',
                   'insured date of birth',
                   'date of birth',
                   'birth date',
                   'dob'],
 'Gender': ['gender', 'sex'],
 'Age': ['age of insured', 'insured age', 'proposer age', 'age'],
 'Customer ID': ['customer identification number',
                 'customer number',
                 'customer code',
                 'customer id',
                 'client id',
                 'client code',
                 'customer ref no',
                 'customer identification no',
                 'client number',
                 'client identification number',
                 'customer reference number'],
 'Residential Address': ['policyholder address',
                         'proposer address',
                         'insured address',
                         'customer address',
                         'residential address',
                         'communication address',
                         'correspondence address',
                         'mailing address',
                         'permanent address',
                         'name and address of insured',
                         'name & address of insured',
                         'insured name and address',
                         'address of insured',
                         'address of proposer',
                         'address of policyholder',
                         'insured correspondence address',
                         'contact address',
                         'address for communication',
                         'address for correspondence'],
 'Pincode': ['customer pincode',
             'insured pincode',
             'postal code',
             'pin code',
             'pincode',
             'pin no',
             'postal pin',
             'postal pin code',
             'insured pin code',
             'customer pin code',
             'zip code'],
 'Policy Start': ['date of commencement of policy',
                  'date of commencement',
                  'commencement date',
                  'policy start date',
                  'policy inception date',
                  'inception date',
                  'effective date',
                  'risk commencement date',
                  'period of insurance from',
                  'policy period from',
                  'policy start',
                  'period from',
                  'valid from',
                  'cover start date',
                  'policy commencement date',
                  'coverage start date',
                  'insurance start date',
                  'risk start date',
                  'cover commencement date',
                  'from date & time',
                  'from date and time'],
 'Policy End': ['policy expiry date',
                'policy expiration date',
                'policy end date',
                'date of expiry',
                'expiry date',
                'expiration date',
                'period of insurance to',
                'policy period to',
                'policy end',
                'valid upto',
                'valid up to',
                'cover end date',
                'policy expiration',
                'coverage end date',
                'insurance expiry date',
                'risk end date',
                'cover expiry date',
                'to date & time',
                'to date and time',
                'valid till',
                'valid until'],
 'Renewal Date': ['annual renewal date',
                  'next renewal date',
                  'renewal date',
                  'renewal due date',
                  'date of renewal',
                  'next policy renewal date'],
 'Premium': ['total premium received towards assured benefit on date of commencement of policy',
             'total premium received',
             'premium amount',
             'annual premium',
             'premium paid',
             'premium'],
 'Gross Premium': ['gross premium',
                   'gross insurance premium',
                   'gross premium amount',
                   'gross premium before tax',
                   'gross premium paid'],
 'Net Premium': ['net premium', 'net insurance premium', 'net premium amount', 'net premium before tax'],
 'GST': ['total gst',
         'gst amount',
         'goods and services tax',
         'gst payable',
         'total gst amount',
         'goods & services tax amount',
         'gst tax amount'],
 'CGST': ['central goods and services tax',
          'cgst amount',
          'cgst',
          'central gst amount',
          'central tax amount',
          'cgst payable',
          'central gst',
          'add cgst'],
 'SGST': ['state goods and services tax',
          'sgst amount',
          'sgst',
          'state gst amount',
          'state tax amount',
          'sgst payable',
          'state gst',
          'add sgst'],
 'IGST': ['integrated goods and services tax',
          'igst amount',
          'igst',
          'integrated gst amount',
          'integrated tax amount',
          'igst payable'],
 'Total Premium': ['total premium payable',
                   'total payable premium',
                   'total premium including tax',
                   'total premium',
                   'total amount payable',
                   'total payable',
                   'premium including gst',
                   'grand total premium',
                   'premium inclusive of tax',
                   'total premium inclusive of gst',
                   'total premium including service tax',
                   'final premium payable',
                   'total amount of premium',
                   'final premium',
                   'total premium for the policy period',
                   'total premium rounded off'],
 'Sum Insured / Assured': ['total sum assured',
                           'total sum insured',
                           'basic sum insured',
                           'base sum insured',
                           'sum insured amount',
                           'sum insured',
                           'sum assured',
                           'coverage amount'],
 'Previous Policy Number': ['previous policy number',
                            'previous policy no',
                            'prev policy no',
                            'expiring policy number',
                            'old policy number',
                            'earlier policy no',
                            'prev. policy no.',
                            'prev. policy',
                            'previous policy no.',
                            'expiring policy no'],
 'Cover Note Number': ['cover note number',
                       'cover note no',
                       'covernote no',
                       'cover note reference',
                       'cover note no.',
                       'covernote no.'],
 'Cover Note Date': ['cover note date',
                     'date of cover note',
                     'covernote date',
                     'cover note dt',
                     'covernote dt'],
 'Policy Issuance Date': ['policy issuance date',
                          'policy issue date',
                          'date of issue',
                          'date of issuance',
                          'issued on',
                          'policy issued on',
                          'document date'],
 'UIN': ['uin',
         'unique identification number',
         'product uin',
         'product unique identification number',
         'unique identification no',
         'unique id',
         'uin number',
         'unique identification no.',
         'uin no',
         'uin no.'],
 'Insured Name': ['insured name',
                  'name of insured',
                  'name of the insured',
                  'named insured',
                  "insured's name",
                  'insured full name'],
 'Insured Code': ['insured code',
                  'insured id',
                  'insured customer code',
                  'insured identification number',
                  'insured person code',
                  "insured's code",
                  'insureds code'],
 'Insured GSTIN': ['insured gstin',
                   'insured gst no',
                   'insured gstin no',
                   'customer gstin',
                   'policyholder gstin',
                   'proposer gstin',
                   'gstin of insured',
                   'gstin of customer',
                   'customer gst number'],
 'Insured PAN': ['insured pan',
                 'insured pan no',
                 'customer pan',
                 'proposer pan number',
                 'policyholder pan'],
 'Registration Address': ['registration address',
                          'vehicle registration address',
                          'registered address of vehicle'],
 'Issue Office Name': ['issue office name',
                       'issuing office name',
                       'policy issuing office name',
                       'name of issuing office',
                       'name of issue office',
                       'issuing branch name',
                       'policy issuing office'],
 'Issue Office Code': ['issue office code',
                       'issuing office code',
                       'policy issuing office code',
                       'office code',
                       'issuing branch code',
                       'branch office code'],
 'Issue Office GSTIN': ['issue office gstin',
                        'issuing office gstin',
                        'office gstin',
                        'branch gstin',
                        'insurer gstin',
                        'company gstin',
                        'gstin of insurer',
                        'company gst no',
                        'company gst number',
                        'fgi gstin number'],
 'Issue Office Address': ['issue office address',
                          'issuing office address',
                          'address of issuing office',
                          'issuing branch address',
                          'policy issuing office address',
                          'office address'],
 'Office Email': ['office email',
                  'office email id',
                  'issuing office email',
                  'branch email',
                  'insurer email',
                  'office e mail'],
 'Office Phone': ['office phone',
                  'office telephone',
                  'issuing office phone',
                  'branch phone number',
                  'office contact number'],
 'Servicing Office': ['policy servicing office', 'servicing office name', 'servicing branch'],
 'IRDAI Registration Number': ['irdai registration number',
                               'irda regn no',
                               'irdai reg no',
                               'irda registration no',
                               'irdai registration no',
                               'irdai registration no.',
                               'irda registration no.'],
 'SAC Code': ['sac code',
              'service accounting code',
              'sac number',
              'services accounting code',
              'accounting code of service',
              'sac'],
 'Basic Premium': ['basic premium', 'base premium', 'basic insurance premium', 'base insurance premium'],
 'Service Tax': ['service tax amount', 'service tax', 'service tax including swachh bharat cess'],
 'Stamp Duty': ['stamp duty', 'stamp duty amount', 'stamp duty paid', 'policy stamp duty'],
 'Receipt Number': ['receipt number',
                    'receipt no',
                    'premium receipt no',
                    'payment receipt number',
                    'receipt no.'],
 'Receipt Date': ['receipt date', 'date of receipt', 'premium receipt date'],
 'Payment Mode': ['payment mode', 'mode of payment', 'payment method'],
 'Fund Transfer Number': ['fund transfer no',
                          'fund transfer number',
                          'transaction reference number',
                          'transaction id',
                          'utr number',
                          'fund transfer no.'],
 'Payment Date': ['payment date', 'date of payment', 'transaction date'],
 'Agent Name': ['agent name', 'name of agent', 'insurance agent name'],
 'Agent Code': ['agent code', 'agent id', 'insurance agent code'],
 'Intermediary Name': ['intermediary name', 'name of intermediary', 'intermediary full name'],
 'Intermediary Code': ['intermediary code', 'intermediary id', 'intermediary number'],
 'Broker Name': ['broker name', 'name of broker', 'insurance broker name'],
 'Broker Code': ['broker code', 'broker id', 'insurance broker code'],
 'Nominee': ['nominee name',
             'name of nominee',
             'nominee',
             'nominee owner driver',
             'name of the nominee'],
 'Nominee Relationship': ['nominee relationship',
                          'relationship of nominee',
                          'nominee relation',
                          'relationship with nominee'],
 'Policy Name': ['product', 'product name', 'insurance product'],
 'Company PAN': ['company pan', 'nia pan number', 'insurer pan number'],
 'Invoice Number': ['invoice no',
                    'tax invoice no',
                    'tax invoice number',
                    'invoice no.',
                    'tax invoice no.'],
 'Product Code': ['product code'],
 'Collection Number': ['collection no', 'collection number', 'collection no.'],
 'Collection Date': ['collection date', 'collection dt'],
 'Development Officer Code': ['dev.off.code', 'dev off code', 'development officer code'],
 'Lead Number': ['lead/breakin no', 'lead breakin no'],
 'Agency Code': ['agency code', 'agency/broker code'],
 'Agency Name': ['agency name'],
 'EMF Code': ['emf code'],
 'Fulfiller Code': ['fulfiller code'],
 'Business Source Code': ['business source code'],
 'Debit Note Number': ['debitnote number', 'debit note no'],
 'Place of Supply': ['place of supply', 'place of supply state code'],
 'Insured State': ['insured state & code', 'insured state'],
 'Special Discount': ['special discount'],
 'Additional Discount': ['additional discount'],
 'Additional Loading': ['additional loading'],
 'Terrorism Surcharge': ['terrorism surcharge'],
 'Highest Sum Insured': ['highest sum insured'],
 'Scope of Cover': ['scope of cover'],
 'Risk Covered': ['risk covered'],
 'Special Exclusions': ['special exclusions'],
 'Special Perils': ['special perils'],
 'Corporate Identification Number': ['corporate identification number',
                                     'cin',
                                     'corporate identification no',
                                     'corporate identification no.'],
 'Channel Name': ['channel name'],
 'Bank RM Employee Code': ['bank rm employee code'],
 'Policy Printed Date': ['policy printed on'],
 'Policy Printed By': ['policy printed by'],
 'Proposal Number': ['proposal no', 'proposal number', 'proposal reference number', 'proposal no.'],
 'Proposal Date': ['date of proposal', 'proposal date', 'date of application'],
 'Proposer Code': ['proposer code', 'proposer id'],
 'Customer Name': ['customer name', 'name of customer'],
 'CKYC Number': ['cust ckyc no', 'ckyc number', 'ckyc no', 'customer ckyc number'],
 'Policy Category': ['policy category', 'category of policy'],
 'Policy Term': ['policy term', 'term of policy', 'policy duration'],
 'First Policy Inception Date': ['date of inception of first policy', 'first policy inception date'],
 'Client Type': ['client type', 'customer type'],
 'Co-insurance Share': ['own share', 'co insurance share'],
 'Financier Details': ['financier(s) details', 'financier details', 'financiers details'],
 'Tax Invoice Number': ['gst invoice no', 'tax invoice no', 'tax invoice number', 'tax invoice no.'],
 'Entered By': ['entered by'],
 'Approved By': ['approved by'],
 'Examined By': ['examined by']}

SPECIFIC = {'Vehicle Insurance': {'Registration Number': ['vehicle registration number',
                                               'registration number',
                                               'registration no',
                                               'vehicle reg no',
                                               'reg no',
                                               'registration mark',
                                               'registration no.',
                                               'vehicle registration no',
                                               'vehicle reg number',
                                               'regn no',
                                               'registration mark number',
                                               'vehicle number',
                                               'vehicle no',
                                               'vehicle no.',
                                               'vehicle registration no.',
                                               'reg no.',
                                               'regn no.'],
                       'Vehicle Make': ['vehicle make',
                                        'make of vehicle',
                                        'manufacturer',
                                        'make',
                                        'make of motor vehicle',
                                        'vehicle manufacturer',
                                        'manufacturer name'],
                       'Vehicle Model': ['vehicle model',
                                         'model of vehicle',
                                         'model',
                                         'model variant',
                                         'model - variant',
                                         'model of motor vehicle',
                                         'vehicle model name',
                                         'model name',
                                         'make/model of vehicle',
                                         'make and model of vehicle insured'],
                       'Engine Number': ['engine number',
                                         'engine no',
                                         'engine no.',
                                         'engine serial number',
                                         'motor engine number'],
                       'Chassis Number': ['chassis number',
                                          'chassis no',
                                          'chassis no.',
                                          'chassis serial number',
                                          'vin',
                                          'vehicle identification number',
                                          'vehicle identification no',
                                          'vehicle identification no.'],
                       'Manufacturing Year': ['year of manufacture',
                                              'manufacturing year',
                                              'mfg year',
                                              'mfg yr',
                                              'mfg. yr',
                                              'mfg. year',
                                              'manufacture year',
                                              'year of mfg',
                                              'year of manufacturing'],
                       'Seating Capacity': ['seating capacity',
                                            'seating including driver',
                                            'seating',
                                            'seats',
                                            'seating capacity including driver',
                                            'seating cap',
                                            'no of seats',
                                            'number of seats'],
                       'Body Type': ['body type', 'type of body'],
                       'IDV': ['total idv',
                               'total insured declared value',
                               'insured declared value',
                               'idv'],
                       'CC': ['cubic capacity',
                              'engine capacity',
                              'cc',
                              'cubic capacity cc',
                              'engine cc',
                              'cc hp',
                              'cubic capacity horsepower'],
                       'Variant': ['vehicle variant', 'model variant name', 'variant'],
                       'RTO': ['rto',
                               'rto location',
                               'registering authority',
                               'registration authority',
                               'rto where vehicle is/will be registered'],
                       'Net Liability Premium': ['net liability premium',
                                                 'net liability amount',
                                                 'net third party liability premium',
                                                 'net liability premium (b)',
                                                 'total liability premium (b)'],
                       'Total Package Premium': ['total package premium',
                                                 'package premium total',
                                                 'package premium',
                                                 'total premium (a+b)',
                                                 'total annual premium (a+b)'],
                       'Basic Third Party Liability': ['basic third party liability',
                                                       'basic tp premium',
                                                       'basic third party premium'],
                       'Own Damage Premium': ['own damage premium',
                                              'basic own damage premium',
                                              'od premium',
                                              'basic od premium',
                                              'basic premium on vehicle',
                                              'basic premium on vehicle and accessories'],
                       'PA Cover Amount': ['pa cover amount',
                                           'personal accident sum insured',
                                           'owner driver pa sum insured'],
                       'PA Cover Premium': ['pa cover premium',
                                            'personal accident premium',
                                            'owner driver pa premium'],
                       'No Claim Bonus': ['no claim bonus', 'no claims bonus', 'ncb', 'ncb percentage'],
                       'Previous Insurer': ['previous insurer',
                                            'previous insurance company',
                                            'name of previous insurer'],
                       'Geographical Area': ['geographical area',
                                             'geographical limits',
                                             'geographic area',
                                             'territorial limits'],
                       'Vehicle IDV': ['vehicle idv',
                                       'idv of vehicle',
                                       'vehicle insured declared value'],
                       'Trailer IDV': ['trailer idv', 'idv of trailer'],
                       'Electrical Accessories IDV': ['electrical accessories idv',
                                                      'idv of electrical accessories'],
                       'Non-electrical Accessories IDV': ['non electrical accessories idv',
                                                          'idv of non electrical accessories'],
                       'CNG/LPG Kit IDV': ['cng lpg kit idv', 'idv of cng lpg kit'],
                       'Vehicle Class': ['class of vehicle'],
                       'Vehicle Usage': ['usage of the vehicle'],
                       'Vehicle Age': ['age of vehicle'],
                       'Area Extension': ['area extension'],
                       'Hypothecation': ['hypothecation agreement with', 'hypothecated to'],
                       'Net Own Damage Premium': ['net own damage premium (a)',
                                                  'total own damage premium (a) (rounded off)'],
                       'Add-on Premium': ['add-on premium', 'add on premium'],
                       'Compulsory Deductible': ['compulsory deductible', 'compulsory excess'],
                       'Voluntary Deductible': ['voluntary deductible'],
                       'Additional Excess': ['additional excess'],
                       'Legal Liability Driver Premium': ['legal liability (wc) to driver (imt-28)',
                                                          'legal liability to paid driver'],
                       'Previous Policy End': ['expiring policy expiry date',
                                               'previous policy expiry date'],
                       'Previous No Claim Bonus': ['no claim bonus % under expiring policy'],
                       'Previous Claim': ['is there any claim in expiring policy'],
                       'Renewal IDV': ['vehicle idv on renewal'],
                       'Nominee Relationship': ['nominee relationship with insured'],
                       'Nominee Age': ['nominee age in y or m', 'nominee age'],
                       'Nominee Share': ['nominee %'],
                       'Appointee Relationship': ['relationship of appointee with nominee'],
                       'Motor Liability Period': ['motor liability period'],
                       'Registration Address': ['registration address of the insured'],
                       'Residential Address': ['communication address of the insured']},
 'Health Insurance': {'Waiting Period': ['initial waiting period', 'waiting period'],
                      'Co-payment': ['co-payment', 'co payment', 'copayment', 'co-pay', 'copay'],
                      'Doctor': ['treating doctor',
                                 'consulting doctor',
                                 'doctor name',
                                 'name of doctor'],
                      'TPA ID': ['tpa id',
                                 'tpa code',
                                 'tpa identification number',
                                 'third party administrator code'],
                      'TPA Name': ['tpa name',
                                   'name of tpa',
                                   'third party administrator name',
                                   'third party administrator',
                                   'name of third party administrator'],
                      'TPA Address': ['tpa address',
                                      'address of tpa',
                                      'third party administrator address'],
                      'TPA Toll Free': ['tpa toll free',
                                        'tpa toll free number',
                                        'tpa toll free no',
                                        'tpa helpline number'],
                      'TPA Phone': ['tpa phone',
                                    'tpa telephone',
                                    'tpa contact number',
                                    'tpa phone number'],
                      'TPA Email': ['tpa email', 'tpa email id', 'tpa e mail'],
                      'Hospital Name': ['hospital name', 'name of hospital', 'network hospital name'],
                      'Room Rent Limit': ['room rent limit',
                                          'room rent capping',
                                          'room rent eligibility'],
                      'Maternity Benefits': ['maternity benefits',
                                             'maternity cover',
                                             'maternity benefit limit',
                                             'maternity benefits opted'],
                      'Number of Members': ['number of members covered',
                                            'number of insured persons',
                                            'total persons covered',
                                            'number of employees covered',
                                            'total members covered'],
                      '9 Months Waiting Period Deleted': ['9 months waiting period deleted',
                                                          '9 month waiting period deleted',
                                                          'deletion of 9 months waiting period'],
                      '30 Days Waiting Period Deleted': ['30 days waiting period deleted',
                                                         '30 day waiting period deleted',
                                                         'deletion of 30 days waiting period'],
                      '2/4 Year Exclusion Deleted': ['2/4 year exclusion deleted',
                                                     '2 4 year exclusion deleted',
                                                     'deletion of 2/4 year exclusion',
                                                     'deletion of 2 4 year exclusion'],
                      'Pre-existing Cover Opted': ['pre existing cover opted',
                                                   'pre-existing disease cover opted',
                                                   'pre existing disease cover opted',
                                                   'pre-existing cover opted'],
                      'Pre-existing Disease': ['pre existing disease', 'pre-existing disease'],
                      'Member ID': ['member id',
                                    'member code',
                                    'insured member id',
                                    'insured person id'],
                      'Normal Delivery Limit': ['normal delivery limit', 'normal delivery'],
                      'Caesarean Limit': ['caesarean limit',
                                          'caesarean delivery',
                                          'caesarian section limit',
                                          'caesarean section limit',
                                          'caesarian section',
                                          'caesarean delivery limit'],
                      'Zone Opted': ['zone opted'],
                      'Ambulance Limit': ['ambulance charges',
                                          'ambulance limit',
                                          'limit of additional ambulance charges per person',
                                          'limit of additional ambulance charges'],
                      'Number of Employees': ['no. of employees / members covered',
                                              'number of employees covered',
                                              'no of employees covered'],
                      'Number of Persons Covered': ['no. of persons covered',
                                                    'number of persons covered',
                                                    'total persons covered'],
                      'Additional Cover Opted': ['additional cover opted', 'additional coverage opted'],
                      'Corporate Buffer': ['corporate buffer', 'corporate buffer limit'],
                      'Installment Facility': ['installment facility optn',
                                               'installment facility option',
                                               'instalment facility option'],
                      'Premium Frequency': ['premium payment frequency', 'frequency of premium payment'],
                      'Installment Amount': ['installment amount rs',
                                             'installment amount',
                                             'instalment amount'],
                      'Bonus': ['bonus', 'cumulative bonus', 'no claim bonus amount'],
                      'Limit of Coverage': ['limit of coverage', 'coverage limit'],
                      'Recharge Benefit': ['recharge benefit', 'recharge amount'],
                      'Family Floater Cover': ['basic floater sum insured',
                                               'family floater sum insured',
                                               'floater sum insured'],
                      'Family Size': ['scheme description (family size)',
                                      'scheme description family size',
                                      'family size'],
                      'Voluntary Co-payment': ['voluntary co-payment', 'voluntary co payment'],
                      'TPA Fax': ['tpa fax', 'tpa fax no'],
                      'Overall Policy Limit': ['overall policy limit'],
                      'Limit per Family': ['limit per family', 'family limit'],
                      'Cover Name': ['name of cover', 'cover name'],
                      'Insured Person': ['name of the insured person',
                                         'name of insured member',
                                         'name of the insured',
                                         'insured member name'],
                      'Relationship': ['relationship to proposer',
                                       'relationship with proposer',
                                       'relationship']},
 'Family Insurance': {'Primary Insured': ['primary insured', 'primary policyholder'],
                      'Family Member': ['family member',
                                        'insured member',
                                        'dependent name',
                                        'name of dependent'],
                      'Relationship': ['relationship to proposer',
                                       'relationship with proposer',
                                       'relationship'],
                      'Family Floater Cover': ['family floater sum insured',
                                               'floater sum insured',
                                               'family floater cover',
                                               'basic floater sum insured'],
                      'Waiting Period': ['waiting period'],
                      'Co-payment': ['co-payment', 'co payment', 'copayment', 'co-pay'],
                      'TPA ID': ['tpa id',
                                 'tpa code',
                                 'tpa identification number',
                                 'third party administrator code'],
                      'TPA Name': ['tpa name',
                                   'name of tpa',
                                   'third party administrator name',
                                   'third party administrator',
                                   'name of third party administrator'],
                      'TPA Address': ['tpa address',
                                      'address of tpa',
                                      'third party administrator address'],
                      'TPA Toll Free': ['tpa toll free',
                                        'tpa toll free number',
                                        'tpa toll free no',
                                        'tpa helpline number'],
                      'TPA Phone': ['tpa phone',
                                    'tpa telephone',
                                    'tpa contact number',
                                    'tpa phone number'],
                      'TPA Email': ['tpa email', 'tpa email id', 'tpa e mail'],
                      'Hospital Name': ['hospital name', 'name of hospital', 'network hospital name'],
                      'Room Rent Limit': ['room rent limit',
                                          'room rent capping',
                                          'room rent eligibility'],
                      'Maternity Benefits': ['maternity benefits',
                                             'maternity cover',
                                             'maternity benefit limit',
                                             'maternity benefits opted'],
                      'Number of Members': ['number of members covered',
                                            'number of insured persons',
                                            'total persons covered',
                                            'number of employees covered',
                                            'total members covered'],
                      '9 Months Waiting Period Deleted': ['9 months waiting period deleted',
                                                          '9 month waiting period deleted',
                                                          'deletion of 9 months waiting period'],
                      '30 Days Waiting Period Deleted': ['30 days waiting period deleted',
                                                         '30 day waiting period deleted',
                                                         'deletion of 30 days waiting period'],
                      '2/4 Year Exclusion Deleted': ['2/4 year exclusion deleted',
                                                     '2 4 year exclusion deleted',
                                                     'deletion of 2/4 year exclusion',
                                                     'deletion of 2 4 year exclusion'],
                      'Pre-existing Cover Opted': ['pre existing cover opted',
                                                   'pre-existing disease cover opted',
                                                   'pre existing disease cover opted',
                                                   'pre-existing cover opted'],
                      'Pre-existing Disease': ['pre existing disease', 'pre-existing disease'],
                      'Member ID': ['member id',
                                    'member code',
                                    'insured member id',
                                    'insured person id'],
                      'Normal Delivery Limit': ['normal delivery limit', 'normal delivery'],
                      'Caesarean Limit': ['caesarean limit',
                                          'caesarean delivery',
                                          'caesarian section limit',
                                          'caesarean section limit',
                                          'caesarian section',
                                          'caesarean delivery limit'],
                      'Zone Opted': ['zone opted'],
                      'Ambulance Limit': ['ambulance charges',
                                          'ambulance limit',
                                          'limit of additional ambulance charges per person',
                                          'limit of additional ambulance charges'],
                      'Number of Employees': ['no. of employees / members covered',
                                              'number of employees covered',
                                              'no of employees covered'],
                      'Number of Persons Covered': ['no. of persons covered',
                                                    'number of persons covered',
                                                    'total persons covered'],
                      'Additional Cover Opted': ['additional cover opted', 'additional coverage opted'],
                      'Corporate Buffer': ['corporate buffer', 'corporate buffer limit'],
                      'Installment Facility': ['installment facility optn',
                                               'installment facility option',
                                               'instalment facility option'],
                      'Premium Frequency': ['premium payment frequency', 'frequency of premium payment'],
                      'Installment Amount': ['installment amount rs',
                                             'installment amount',
                                             'instalment amount'],
                      'Bonus': ['bonus', 'cumulative bonus', 'no claim bonus amount'],
                      'Limit of Coverage': ['limit of coverage', 'coverage limit'],
                      'Recharge Benefit': ['recharge benefit', 'recharge amount'],
                      'Family Size': ['scheme description (family size)',
                                      'scheme description family size',
                                      'family size'],
                      'Voluntary Co-payment': ['voluntary co-payment', 'voluntary co payment'],
                      'TPA Fax': ['tpa fax', 'tpa fax no'],
                      'Overall Policy Limit': ['overall policy limit'],
                      'Limit per Family': ['limit per family', 'family limit'],
                      'Cover Name': ['name of cover', 'cover name'],
                      'Insured Person': ['name of the insured person',
                                         'name of insured member',
                                         'name of the insured',
                                         'insured member name']},
 'Life Insurance': {'Life Assured': ['name of life assured', 'life assured name', 'life assured'],
                    'Nominee': ['name of nominee', 'nominee name', 'nominee'],
                    'Death Benefit': ['death benefit', 'sum assured on death'],
                    'Premium Frequency': ['frequency of premium payment',
                                          'premium payment frequency',
                                          'premium payment mode',
                                          'premium mode',
                                          'premium frequency',
                                          'frequency of payment',
                                          'mode of premium payment',
                                          'premium paying frequency'],
                    'Number of Members': ['number of members covered as on the date of commencement of '
                                          'master policy',
                                          'number of members covered',
                                          'number of members',
                                          'total eligible members',
                                          'total members covered',
                                          'number of lives covered',
                                          'total insured members'],
                    'Maturity Date': ['maturity date', 'date of maturity', 'policy maturity date'],
                    'Appointee': ['appointee name', 'name of appointee', 'appointee'],
                    'Policy Term': ['policy term', 'term of policy', 'policy duration'],
                    'Premium Paying Term': ['premium paying term',
                                            'premium payment term',
                                            'premium payment duration'],
                    'Master Policy Number': ['master policy number',
                                             'master policy no',
                                             'master policy no.'],
                    'Scheme Name': ['name of the scheme', 'scheme name'],
                    'Group Type': ['type of group', 'group type'],
                    'Divisional Office': ['divisional office'],
                    'PGS Unit Office': ['p&gs unit office', 'pgs unit office'],
                    'Minimum Entry Age': ['minimum entry age', 'min entry age'],
                    'Maximum Entry Age': ['maximum entry age', 'max entry age'],
                    'Maximum Cover Ceasing Age': ['maximum cover ceasing age', 'cover ceasing age'],
                    'Premium Due Date': ['premium due date', 'next premium due date'],
                    'Rider Opted': ['rider opted', 'details of rider opted'],
                    'Rider Sum Assured': ['rider sum assured']},
 'Others': {'Number of Members': ['number of members covered', 'total members covered'],
            'Accident Cover': ['personal accident cover', 'accident cover'],
            'Number of Students': ['no of students', 'no. of students', 'number of students'],
            'Number of Parents': ['no of parents', 'number of parents'],
            'Number of Lives': ['no.of lives', 'no of lives', 'number of lives'],
            'Medical Expenses per Student': ['medical expenses per student (inclusive of opd)',
                                             'medical expenses per student'],
            'Limit per Student': ['limit per student'],
            'Limit per Accident': ['limit per accident'],
            'Parents Sum Insured': ['total si of parents or guardian for payment of tuition and hostel '
                                    'fees'],
            'Tuition and Hostel Fees': ['payment of tuition and hostel fee for remaining semesters'],
            'Risk Group': ['risk group', 'risk class'],
            'Accidental Death Cover': ['accidental death cover', 'accidental death benefit'],
            'Permanent Total Disablement': ['permanent total disablement', 'permanent total disability'],
            'Permanent Partial Disablement': ['permanent partial disablement',
                                              'permanent partial disability'],
            'Cover Name': ['cover name', 'name of cover'],
            'Cover Definition': ['cover definition'],
            'Sum Insured Basis': ['sum insured (basis)', 'sum insured basis'],
            'Cremation Charges': ['cremation charges'],
            'Repatriation Benefit': ['repatriation of mortal remains', 'repatriation of remains']}}

PROPERTY_FIELDS = {'Bank / Financial Institution': ['name of bank financial institution',
                                  'name of bank/financial institution',
                                  'name of bank',
                                  'financial institution',
                                  'hypothecated to',
                                  'mortgagee'],
 'Location of Risk': ['location of the risk premises',
                      'location of the risk / premises',
                      'location of risk',
                      'risk location',
                      'risk address',
                      'insured premises'],
 'Occupation': ['occupied as', 'nature of occupancy', 'occupancy', 'occupation'],
 'Terrorism Premium': ['terrorism premium', 'add terrorism', 'terrorism'],
 'Description of Property': ['description of property',
                             'property description',
                             'description of insured property'],
 'Building Coverage': ['building sum insured', 'sum insured building', 'building coverage amount'],
 'Machinery Coverage': ['machinery sum insured',
                        'plant and machinery sum insured',
                        'machinery coverage amount'],
 'Excess': ['policy excess', 'deductible amount', 'excess amount', 'deductible'],
 'Special Conditions': ['special conditions',
                        'special policy conditions',
                        'special conditions / clauses / warranties applicable if any'],
 'Clauses': ['applicable clauses', 'policy clauses', 'clauses'],
 'Warranties': ['applicable warranties', 'policy warranties', 'warranties'],
 'Issued At': ['issued at', 'place of issue'],
 'Escalation Clause': ['escalation clause']}

ALL_ALIASES = sorted(set(ALL_ALIASES).union(*(set(v) for s in [COMMON, PROPERTY_FIELDS, *SPECIFIC.values()] for v in s.values())), key=len, reverse=True)

# Explicit aliases for common schedule labels; generic labels are resolved
# only inside their owning section by evidence_upgrade.py.
_extend_schema(COMMON, {
 'Mobile Number':'mob no|mob number|mobile phone no|cell no|cell phone no|insured mobile|customer mobile|contact mobile no|mobile contact number',
 'Phone Number':'tel|tel no.|phone no.|telephone no.|contact phone|insured telephone|insured phone|customer contact no',
 'Email ID':'e-mail/fax|email/fax|e mail fax|e-mail address|email id of insured|insured e-mail|customer e-mail',
 'Residential Address':'insured premises address|address of the insured|insured mailing address|insured communication address|postal address of insured|address of customer',
 'Insured Name':"insured's name|name of the insured person|name of insured company|insured company name",
 'Office Email':'issuing office e-mail|office e-mail id|branch e-mail',
 'Issue Office Code':'issuing office details code|office code no',
})
ALL_ALIASES = sorted(set(ALL_ALIASES).union(*(set(v) for s in [COMMON, PROPERTY_FIELDS, *SPECIFIC.values()] for v in s.values())), key=len, reverse=True)
_previous_validation = validate_field
def validate_field(field, value):
    if field == 'Phone Number' and re.fullmatch(r'[/ ]*[Xx*]{4,10}\d{2,6}', str(value).strip()):
        return True  # Preserve the printed mask; never reconstruct hidden digits.
    return _previous_validation(field, value)


# Focused active schema; legacy aliases remain as extraction boundaries.
COMMON = {'Insurer': ['name of insurer', 'name of insurance company', 'insurance company', 'insurer'],
 'Policy Number': ['policy certificate number',
                   'insurance policy number',
                   'policy reference number',
                   'policy number',
                   'policy no',
                   'policy id',
                   'policy ref',
                   'policy no.',
                   'policy reference no',
                   'current policy number',
                   'current policy no',
                   'policy certificate no',
                   'policy document number',
                   'policy #',
                   'policy serial number',
                   'policy certificate no.',
                   'insurance policy no',
                   'policy document no',
                   'policy serial no'],
 'Policy Name': ['product', 'product name', 'insurance product'],
 'UIN': ['uin',
         'unique identification number',
         'product uin',
         'product unique identification number',
         'unique identification no',
         'unique id',
         'uin number',
         'unique identification no.',
         'uin no',
         'uin no.',
         'product unique identification no'],
 'Policyholder': ['name of the master policyholder',
                  'name of master policyholder',
                  'name of policyholder',
                  'name of policy holder',
                  'policyholder name',
                  'policy holder name',
                  'name and address of policyholder',
                  'policyholder',
                  'policy holder',
                  'master policyholder name',
                  'name of policy owner',
                  'policy owner name'],
 'Insured Name': ['insured name',
                  'name of insured',
                  'name of the insured',
                  'named insured',
                  "insured's name",
                  'insured full name',
                  'name of insured company',
                  'insured company name'],
 'Proposer Name': ['name of the proposer',
                   'name of proposer',
                   'proposer name',
                   "proposer's name",
                   'proposer',
                   'applicant name',
                   'name of applicant',
                   'proposer full name'],
 'Insured Person': ['name of insured person',
                    'insured person name',
                    'name of insured member',
                    'insured member name',
                    'insured person'],
 'Customer ID': ['customer identification number',
                 'customer number',
                 'customer code',
                 'customer id',
                 'client id',
                 'client code',
                 'customer ref no',
                 'customer identification no',
                 'client number',
                 'client identification number',
                 'customer reference number',
                 'customer no',
                 'client no',
                 'client identification no',
                 'customer reference no'],
 'Insured Code': ['insured code',
                  'insured id',
                  'insured customer code',
                  'insured identification number',
                  'insured person code',
                  "insured's code",
                  'insureds code',
                  'insured identification no'],
 'Mobile Number': ['policyholder mobile number',
                   'proposer mobile number',
                   'insured mobile number',
                   'customer mobile number',
                   'registered mobile number',
                   'mobile number',
                   'mobile no',
                   'mobile',
                   'mobile no.',
                   'mobile telephone',
                   'cell number',
                   'cell phone number',
                   'insured mobile no',
                   'customer mobile no',
                   'proposer mobile no',
                   'registered mobile',
                   'mobile phone',
                   'mob no',
                   'mob number',
                   'mobile phone no',
                   'cell no',
                   'cell phone no',
                   'insured mobile',
                   'customer mobile',
                   'contact mobile no',
                   'mobile contact number',
                   'policyholder mobile no',
                   'registered mobile no',
                   'mobile contact no'],
 'Phone Number': ['policyholder phone number',
                  'insured contact number',
                  'customer phone number',
                  'proposer contact number',
                  'contact number',
                  'contact no',
                  'phone number',
                  'phone no',
                  'telephone number',
                  'landline number',
                  'telephone no',
                  'tel no',
                  'landline no',
                  'insured phone no',
                  'customer telephone number',
                  'phone',
                  'telephone',
                  'contact no.',
                  'phone no.',
                  'tel',
                  'tel no.',
                  'telephone no.',
                  'contact phone',
                  'insured telephone',
                  'insured phone',
                  'customer contact no',
                  'policyholder phone no',
                  'insured contact no',
                  'customer phone no',
                  'proposer contact no',
                  'customer telephone no'],
 'Email ID': ['policyholder email',
              'proposer email id',
              'proposer email',
              'insured email',
              'customer email',
              'registered email id',
              'email address',
              'email id',
              'e-mail id',
              'e-mail',
              'email',
              'e mail',
              'e mail id',
              'e mail address',
              'insured email id',
              'customer email id',
              'policyholder email id',
              'proposer email address',
              'e-mail/fax',
              'email/fax',
              'e mail fax',
              'e-mail address',
              'email id of insured',
              'insured e-mail',
              'customer e-mail'],
 'Residential Address': ['policyholder address',
                         'proposer address',
                         'insured address',
                         'customer address',
                         'residential address',
                         'communication address',
                         'correspondence address',
                         'mailing address',
                         'permanent address',
                         'name and address of insured',
                         'name & address of insured',
                         'insured name and address',
                         'address of insured',
                         'address of proposer',
                         'address of policyholder',
                         'insured correspondence address',
                         'contact address',
                         'address for communication',
                         'address for correspondence',
                         'insured premises address',
                         'address of the insured',
                         'insured mailing address',
                         'insured communication address',
                         'postal address of insured',
                         'address of customer'],
 'Pincode': ['customer pincode',
             'insured pincode',
             'postal code',
             'pin code',
             'pincode',
             'pin no',
             'postal pin',
             'postal pin code',
             'insured pin code',
             'customer pin code',
             'zip code'],
 'Date of Birth': ['proposer date of birth',
                   'insured date of birth',
                   'date of birth',
                   'birth date',
                   'dob',
                   'proposer dt of birth',
                   'insured dt of birth',
                   'dt of birth',
                   'birth dt'],
 'Gender': ['gender', 'sex'],
 'Age': ['age of insured', 'insured age', 'proposer age', 'age'],
 'Insured GSTIN': ['insured gstin',
                   'insured gst no',
                   'insured gstin no',
                   'customer gstin',
                   'policyholder gstin',
                   'proposer gstin',
                   'gstin of insured',
                   'gstin of customer',
                   'customer gst number',
                   'customer gst no'],
 'Insured PAN': ['insured pan',
                 'insured pan no',
                 'customer pan',
                 'proposer pan number',
                 'policyholder pan',
                 'proposer pan no'],
 'Policy Start': ['date of commencement of policy',
                  'date of commencement',
                  'commencement date',
                  'policy start date',
                  'policy inception date',
                  'inception date',
                  'effective date',
                  'risk commencement date',
                  'period of insurance from',
                  'policy period from',
                  'policy start',
                  'period from',
                  'valid from',
                  'cover start date',
                  'policy commencement date',
                  'coverage start date',
                  'insurance start date',
                  'risk start date',
                  'cover commencement date',
                  'from date & time',
                  'from date and time',
                  'dt of commencement of policy',
                  'dt of commencement',
                  'commencement dt',
                  'policy start dt',
                  'policy inception dt',
                  'inception dt',
                  'effective dt',
                  'risk commencement dt',
                  'cover start dt',
                  'policy commencement dt',
                  'coverage start dt',
                  'insurance start dt',
                  'risk start dt',
                  'cover commencement dt',
                  'from dt & time',
                  'from dt and time'],
 'Policy End': ['policy expiry date',
                'policy expiration date',
                'policy end date',
                'date of expiry',
                'expiry date',
                'expiration date',
                'period of insurance to',
                'policy period to',
                'policy end',
                'valid upto',
                'valid up to',
                'cover end date',
                'policy expiration',
                'coverage end date',
                'insurance expiry date',
                'risk end date',
                'cover expiry date',
                'to date & time',
                'to date and time',
                'valid till',
                'valid until',
                'policy expiry dt',
                'policy expiration dt',
                'policy end dt',
                'dt of expiry',
                'expiry dt',
                'expiration dt',
                'cover end dt',
                'coverage end dt',
                'insurance expiry dt',
                'risk end dt',
                'cover expiry dt',
                'to dt & time',
                'to dt and time'],
 'Renewal Date': ['annual renewal date',
                  'next renewal date',
                  'renewal date',
                  'renewal due date',
                  'date of renewal',
                  'next policy renewal date',
                  'annual renewal dt',
                  'next renewal dt',
                  'renewal dt',
                  'renewal due dt',
                  'dt of renewal',
                  'next policy renewal dt'],
 'Policy Issuance Date': ['policy issuance date',
                          'policy issue date',
                          'date of issue',
                          'date of issuance',
                          'issued on',
                          'policy issued on',
                          'document date',
                          'policy issuance dt',
                          'policy issue dt',
                          'dt of issue',
                          'dt of issuance',
                          'document dt'],
 'Previous Policy Number': ['previous policy number',
                            'previous policy no',
                            'prev policy no',
                            'expiring policy number',
                            'old policy number',
                            'earlier policy no',
                            'prev. policy no.',
                            'prev. policy',
                            'previous policy no.',
                            'expiring policy no',
                            'old policy no'],
 'Proposal Number': ['proposal no',
                     'proposal number',
                     'proposal reference number',
                     'proposal no.',
                     'proposal reference no'],
 'Proposal Date': ['date of proposal',
                   'proposal date',
                   'date of application',
                   'dt of proposal',
                   'proposal dt',
                   'dt of application'],
 'Issue Office Name': ['issue office name',
                       'issuing office name',
                       'policy issuing office name',
                       'name of issuing office',
                       'name of issue office',
                       'issuing branch name',
                       'policy issuing office'],
 'Issue Office Code': ['issue office code',
                       'issuing office code',
                       'policy issuing office code',
                       'office code',
                       'issuing branch code',
                       'branch office code',
                       'issuing office details code',
                       'office code no'],
 'Issue Office Address': ['issue office address',
                          'issuing office address',
                          'address of issuing office',
                          'issuing branch address',
                          'policy issuing office address',
                          'office address'],
 'Issue Office GSTIN': ['issue office gstin',
                        'issuing office gstin',
                        'office gstin',
                        'branch gstin',
                        'insurer gstin',
                        'company gstin',
                        'gstin of insurer',
                        'company gst no',
                        'company gst number',
                        'fgi gstin number',
                        'fgi gstin no'],
 'Office Email': ['office email',
                  'office email id',
                  'issuing office email',
                  'branch email',
                  'insurer email',
                  'office e mail',
                  'issuing office e-mail',
                  'office e-mail id',
                  'branch e-mail'],
 'Office Phone': ['office phone',
                  'office telephone',
                  'issuing office phone',
                  'branch phone number',
                  'office contact number',
                  'branch phone no',
                  'office contact no'],
 'Premium': ['total premium received towards assured benefit on date of commencement of policy',
             'total premium received',
             'premium amount',
             'annual premium',
             'premium paid',
             'premium',
             'total premium received towards assured benefit on dt of commencement of policy'],
 'Basic Premium': ['basic premium', 'base premium', 'basic insurance premium', 'base insurance premium'],
 'Gross Premium': ['gross premium',
                   'gross insurance premium',
                   'gross premium amount',
                   'gross premium before tax',
                   'gross premium paid'],
 'Net Premium': ['net premium', 'net insurance premium', 'net premium amount', 'net premium before tax'],
 'GST': ['total gst',
         'gst amount',
         'goods and services tax',
         'gst payable',
         'total gst amount',
         'goods & services tax amount',
         'gst tax amount'],
 'CGST': ['central goods and services tax',
          'cgst amount',
          'cgst',
          'central gst amount',
          'central tax amount',
          'cgst payable',
          'central gst',
          'add cgst'],
 'SGST': ['state goods and services tax',
          'sgst amount',
          'sgst',
          'state gst amount',
          'state tax amount',
          'sgst payable',
          'state gst',
          'add sgst'],
 'IGST': ['integrated goods and services tax',
          'igst amount',
          'igst',
          'integrated gst amount',
          'integrated tax amount',
          'igst payable'],
 'Service Tax': ['service tax amount', 'service tax', 'service tax including swachh bharat cess'],
 'Stamp Duty': ['stamp duty', 'stamp duty amount', 'stamp duty paid', 'policy stamp duty'],
 'Total Premium': ['total premium payable',
                   'total payable premium',
                   'total premium including tax',
                   'total premium',
                   'total amount payable',
                   'total payable',
                   'premium including gst',
                   'grand total premium',
                   'premium inclusive of tax',
                   'total premium inclusive of gst',
                   'total premium including service tax',
                   'final premium payable',
                   'total amount of premium',
                   'final premium',
                   'total premium for the policy period',
                   'total premium rounded off'],
 'Sum Insured / Assured': ['total sum assured',
                           'total sum insured',
                           'basic sum insured',
                           'base sum insured',
                           'sum insured amount',
                           'sum insured',
                           'sum assured',
                           'coverage amount'],
 'Receipt Number': ['receipt number',
                    'receipt no',
                    'premium receipt no',
                    'payment receipt number',
                    'receipt no.',
                    'payment receipt no'],
 'Receipt Date': ['receipt date',
                  'date of receipt',
                  'premium receipt date',
                  'receipt dt',
                  'dt of receipt',
                  'premium receipt dt'],
 'Payment Mode': ['payment mode', 'mode of payment', 'payment method'],
 'Intermediary Name': ['intermediary name', 'name of intermediary', 'intermediary full name'],
 'Intermediary Code': ['intermediary code', 'intermediary id', 'intermediary number', 'intermediary no']}
SPECIFIC = {'Vehicle Insurance': {'Registration Number': ['vehicle registration number',
                                               'registration number',
                                               'registration no',
                                               'vehicle reg no',
                                               'reg no',
                                               'registration mark',
                                               'registration no.',
                                               'vehicle registration no',
                                               'vehicle reg number',
                                               'regn no',
                                               'registration mark number',
                                               'vehicle number',
                                               'vehicle no',
                                               'vehicle no.',
                                               'vehicle registration no.',
                                               'reg no.',
                                               'regn no.',
                                               'registration mark no'],
                       'Vehicle Make': ['vehicle make',
                                        'make of vehicle',
                                        'manufacturer',
                                        'make',
                                        'make of motor vehicle',
                                        'vehicle manufacturer',
                                        'manufacturer name'],
                       'Vehicle Model': ['vehicle model',
                                         'model of vehicle',
                                         'model',
                                         'model variant',
                                         'model - variant',
                                         'model of motor vehicle',
                                         'vehicle model name',
                                         'model name',
                                         'make/model of vehicle',
                                         'make and model of vehicle insured'],
                       'Engine Number': ['engine number',
                                         'engine no',
                                         'engine no.',
                                         'engine serial number',
                                         'motor engine number',
                                         'engine serial no',
                                         'motor engine no'],
                       'Chassis Number': ['chassis number',
                                          'chassis no',
                                          'chassis no.',
                                          'chassis serial number',
                                          'vin',
                                          'vehicle identification number',
                                          'vehicle identification no',
                                          'vehicle identification no.',
                                          'chassis serial no'],
                       'Manufacturing Year': ['year of manufacture',
                                              'manufacturing year',
                                              'mfg year',
                                              'mfg yr',
                                              'mfg. yr',
                                              'mfg. year',
                                              'manufacture year',
                                              'year of mfg',
                                              'year of manufacturing'],
                       'Seating Capacity': ['seating capacity',
                                            'seating including driver',
                                            'seating',
                                            'seats',
                                            'seating capacity including driver',
                                            'seating cap',
                                            'no of seats',
                                            'number of seats'],
                       'Body Type': ['body type', 'type of body'],
                       'IDV': ['total idv',
                               'total insured declared value',
                               'insured declared value',
                               'idv'],
                       'CC': ['cubic capacity',
                              'engine capacity',
                              'cc',
                              'cubic capacity cc',
                              'engine cc',
                              'cc hp',
                              'cubic capacity horsepower'],
                       'Variant': ['vehicle variant', 'model variant name', 'variant'],
                       'RTO': ['rto',
                               'rto location',
                               'registering authority',
                               'registration authority',
                               'rto where vehicle is/will be registered'],
                       'Net Liability Premium': ['net liability premium',
                                                 'net liability amount',
                                                 'net third party liability premium',
                                                 'net liability premium (b)',
                                                 'total liability premium (b)'],
                       'Total Package Premium': ['total package premium',
                                                 'package premium total',
                                                 'package premium',
                                                 'total premium (a+b)',
                                                 'total annual premium (a+b)'],
                       'Basic Third Party Liability': ['basic third party liability',
                                                       'basic tp premium',
                                                       'basic third party premium'],
                       'Own Damage Premium': ['own damage premium',
                                              'basic own damage premium',
                                              'od premium',
                                              'basic od premium',
                                              'basic premium on vehicle',
                                              'basic premium on vehicle and accessories'],
                       'PA Cover Amount': ['pa cover amount',
                                           'personal accident sum insured',
                                           'owner driver pa sum insured'],
                       'PA Cover Premium': ['pa cover premium',
                                            'personal accident premium',
                                            'owner driver pa premium'],
                       'No Claim Bonus': ['no claim bonus', 'no claims bonus', 'ncb', 'ncb percentage'],
                       'Previous Insurer': ['previous insurer',
                                            'previous insurance company',
                                            'name of previous insurer'],
                       'Geographical Area': ['geographical area',
                                             'geographical limits',
                                             'geographic area',
                                             'territorial limits'],
                       'Vehicle IDV': ['vehicle idv',
                                       'idv of vehicle',
                                       'vehicle insured declared value'],
                       'Trailer IDV': ['trailer idv', 'idv of trailer'],
                       'Electrical Accessories IDV': ['electrical accessories idv',
                                                      'idv of electrical accessories'],
                       'Non-electrical Accessories IDV': ['non electrical accessories idv',
                                                          'idv of non electrical accessories'],
                       'CNG/LPG Kit IDV': ['cng lpg kit idv', 'idv of cng lpg kit'],
                       'Vehicle Class': ['class of vehicle'],
                       'Vehicle Usage': ['usage of the vehicle'],
                       'Vehicle Age': ['age of vehicle'],
                       'Area Extension': ['area extension'],
                       'Hypothecation': ['hypothecation agreement with', 'hypothecated to'],
                       'Net Own Damage Premium': ['net own damage premium (a)',
                                                  'total own damage premium (a) (rounded off)'],
                       'Add-on Premium': ['add-on premium', 'add on premium'],
                       'Compulsory Deductible': ['compulsory deductible', 'compulsory excess'],
                       'Voluntary Deductible': ['voluntary deductible'],
                       'Additional Excess': ['additional excess'],
                       'Legal Liability Driver Premium': ['legal liability (wc) to driver (imt-28)',
                                                          'legal liability to paid driver'],
                       'Previous Policy End': ['expiring policy expiry date',
                                               'previous policy expiry date',
                                               'expiring policy expiry dt',
                                               'previous policy expiry dt'],
                       'Previous No Claim Bonus': ['no claim bonus % under expiring policy'],
                       'Previous Claim': ['is there any claim in expiring policy'],
                       'Renewal IDV': ['vehicle idv on renewal'],
                       'Nominee Relationship': ['nominee relationship with insured'],
                       'Nominee Age': ['nominee age in y or m', 'nominee age'],
                       'Nominee Share': ['nominee %'],
                       'Appointee Relationship': ['relationship of appointee with nominee'],
                       'Motor Liability Period': ['motor liability period'],
                       'Registration Address': ['registration address',
                                                'vehicle registration address',
                                                'registered address of vehicle'],
                       'Nominee': ['nominee name',
                                   'name of nominee',
                                   'nominee',
                                   'nominee owner driver',
                                   'name of the nominee'],
                       'Fund Transfer Number': ['fund transfer no',
                                                'fund transfer number',
                                                'transaction reference number',
                                                'transaction id',
                                                'utr number',
                                                'fund transfer no.',
                                                'transaction reference no',
                                                'utr no'],
                       'Payment Date': ['payment date',
                                        'date of payment',
                                        'transaction date',
                                        'payment dt',
                                        'dt of payment',
                                        'transaction dt'],
                       'Financier Details': ['financier(s) details',
                                             'financier details',
                                             'financiers details']},
 'Health Insurance': {'Waiting Period': ['initial waiting period', 'waiting period'],
                      'Co-payment': ['co-payment', 'co payment', 'copayment', 'co-pay', 'copay'],
                      'Doctor': ['treating doctor',
                                 'consulting doctor',
                                 'doctor name',
                                 'name of doctor'],
                      'TPA ID': ['tpa id',
                                 'tpa code',
                                 'tpa identification number',
                                 'third party administrator code',
                                 'tpa identification no'],
                      'TPA Name': ['tpa name',
                                   'name of tpa',
                                   'third party administrator name',
                                   'third party administrator',
                                   'name of third party administrator'],
                      'TPA Address': ['tpa address',
                                      'address of tpa',
                                      'third party administrator address'],
                      'TPA Toll Free': ['tpa toll free',
                                        'tpa toll free number',
                                        'tpa toll free no',
                                        'tpa helpline number',
                                        'tpa helpline no'],
                      'TPA Phone': ['tpa phone',
                                    'tpa telephone',
                                    'tpa contact number',
                                    'tpa phone number',
                                    'tpa contact no',
                                    'tpa phone no'],
                      'TPA Email': ['tpa email', 'tpa email id', 'tpa e mail'],
                      'Hospital Name': ['hospital name', 'name of hospital', 'network hospital name'],
                      'Room Rent Limit': ['room rent limit',
                                          'room rent capping',
                                          'room rent eligibility'],
                      'Maternity Benefits': ['maternity benefits',
                                             'maternity cover',
                                             'maternity benefit limit',
                                             'maternity benefits opted'],
                      'Number of Members': ['number of members covered',
                                            'number of insured persons',
                                            'total persons covered',
                                            'number of employees covered',
                                            'total members covered',
                                            'no of members covered',
                                            'no of insured persons',
                                            'no of employees covered'],
                      '9 Months Waiting Period Deleted': ['9 months waiting period deleted',
                                                          '9 month waiting period deleted',
                                                          'deletion of 9 months waiting period'],
                      '30 Days Waiting Period Deleted': ['30 days waiting period deleted',
                                                         '30 day waiting period deleted',
                                                         'deletion of 30 days waiting period'],
                      '2/4 Year Exclusion Deleted': ['2/4 year exclusion deleted',
                                                     '2 4 year exclusion deleted',
                                                     'deletion of 2/4 year exclusion',
                                                     'deletion of 2 4 year exclusion'],
                      'Pre-existing Cover Opted': ['pre existing cover opted',
                                                   'pre-existing disease cover opted',
                                                   'pre existing disease cover opted',
                                                   'pre-existing cover opted'],
                      'Pre-existing Disease': ['pre existing disease', 'pre-existing disease'],
                      'Member ID': ['member id',
                                    'member code',
                                    'insured member id',
                                    'insured person id'],
                      'Normal Delivery Limit': ['normal delivery limit', 'normal delivery'],
                      'Caesarean Limit': ['caesarean limit',
                                          'caesarean delivery',
                                          'caesarian section limit',
                                          'caesarean section limit',
                                          'caesarian section',
                                          'caesarean delivery limit'],
                      'Zone Opted': ['zone opted'],
                      'Ambulance Limit': ['ambulance charges',
                                          'ambulance limit',
                                          'limit of additional ambulance charges per person',
                                          'limit of additional ambulance charges'],
                      'Number of Employees': ['no. of employees / members covered',
                                              'number of employees covered',
                                              'no of employees covered'],
                      'Number of Persons Covered': ['no. of persons covered',
                                                    'number of persons covered',
                                                    'total persons covered',
                                                    'no of persons covered'],
                      'Additional Cover Opted': ['additional cover opted', 'additional coverage opted'],
                      'Corporate Buffer': ['corporate buffer', 'corporate buffer limit'],
                      'Installment Facility': ['installment facility optn',
                                               'installment facility option',
                                               'instalment facility option'],
                      'Premium Frequency': ['premium payment frequency', 'frequency of premium payment'],
                      'Installment Amount': ['installment amount rs',
                                             'installment amount',
                                             'instalment amount'],
                      'Bonus': ['bonus', 'cumulative bonus', 'no claim bonus amount'],
                      'Limit of Coverage': ['limit of coverage', 'coverage limit'],
                      'Recharge Benefit': ['recharge benefit', 'recharge amount'],
                      'Family Floater Cover': ['basic floater sum insured',
                                               'family floater sum insured',
                                               'floater sum insured'],
                      'Family Size': ['scheme description (family size)',
                                      'scheme description family size',
                                      'family size'],
                      'Voluntary Co-payment': ['voluntary co-payment', 'voluntary co payment'],
                      'TPA Fax': ['tpa fax', 'tpa fax no'],
                      'Overall Policy Limit': ['overall policy limit'],
                      'Limit per Family': ['limit per family', 'family limit'],
                      'Cover Name': ['name of cover', 'cover name'],
                      'Relationship': ['relationship to proposer',
                                       'relationship with proposer',
                                       'relationship'],
                      'Client Type': ['client type', 'customer type'],
                      'Policy Term': ['policy term', 'term of policy', 'policy duration'],
                      'Broker Name': ['broker name', 'name of broker', 'insurance broker name'],
                      'Broker Code': ['broker code', 'broker id', 'insurance broker code'],
                      'SAC Code': ['sac code',
                                   'service accounting code',
                                   'sac number',
                                   'services accounting code',
                                   'accounting code of service',
                                   'sac',
                                   'sac no']},
 'Family Insurance': {'Primary Insured': ['primary insured', 'primary policyholder'],
                      'Family Member': ['family member',
                                        'insured member',
                                        'dependent name',
                                        'name of dependent'],
                      'Relationship': ['relationship to proposer',
                                       'relationship with proposer',
                                       'relationship'],
                      'Family Floater Cover': ['family floater sum insured',
                                               'floater sum insured',
                                               'family floater cover',
                                               'basic floater sum insured'],
                      'Waiting Period': ['waiting period'],
                      'Co-payment': ['co-payment', 'co payment', 'copayment', 'co-pay'],
                      'TPA ID': ['tpa id',
                                 'tpa code',
                                 'tpa identification number',
                                 'third party administrator code',
                                 'tpa identification no'],
                      'TPA Name': ['tpa name',
                                   'name of tpa',
                                   'third party administrator name',
                                   'third party administrator',
                                   'name of third party administrator'],
                      'TPA Address': ['tpa address',
                                      'address of tpa',
                                      'third party administrator address'],
                      'TPA Toll Free': ['tpa toll free',
                                        'tpa toll free number',
                                        'tpa toll free no',
                                        'tpa helpline number',
                                        'tpa helpline no'],
                      'TPA Phone': ['tpa phone',
                                    'tpa telephone',
                                    'tpa contact number',
                                    'tpa phone number',
                                    'tpa contact no',
                                    'tpa phone no'],
                      'TPA Email': ['tpa email', 'tpa email id', 'tpa e mail'],
                      'Hospital Name': ['hospital name', 'name of hospital', 'network hospital name'],
                      'Room Rent Limit': ['room rent limit',
                                          'room rent capping',
                                          'room rent eligibility'],
                      'Maternity Benefits': ['maternity benefits',
                                             'maternity cover',
                                             'maternity benefit limit',
                                             'maternity benefits opted'],
                      'Number of Members': ['number of members covered',
                                            'number of insured persons',
                                            'total persons covered',
                                            'number of employees covered',
                                            'total members covered',
                                            'no of members covered',
                                            'no of insured persons',
                                            'no of employees covered'],
                      '9 Months Waiting Period Deleted': ['9 months waiting period deleted',
                                                          '9 month waiting period deleted',
                                                          'deletion of 9 months waiting period'],
                      '30 Days Waiting Period Deleted': ['30 days waiting period deleted',
                                                         '30 day waiting period deleted',
                                                         'deletion of 30 days waiting period'],
                      '2/4 Year Exclusion Deleted': ['2/4 year exclusion deleted',
                                                     '2 4 year exclusion deleted',
                                                     'deletion of 2/4 year exclusion',
                                                     'deletion of 2 4 year exclusion'],
                      'Pre-existing Cover Opted': ['pre existing cover opted',
                                                   'pre-existing disease cover opted',
                                                   'pre existing disease cover opted',
                                                   'pre-existing cover opted'],
                      'Pre-existing Disease': ['pre existing disease', 'pre-existing disease'],
                      'Member ID': ['member id',
                                    'member code',
                                    'insured member id',
                                    'insured person id'],
                      'Normal Delivery Limit': ['normal delivery limit', 'normal delivery'],
                      'Caesarean Limit': ['caesarean limit',
                                          'caesarean delivery',
                                          'caesarian section limit',
                                          'caesarean section limit',
                                          'caesarian section',
                                          'caesarean delivery limit'],
                      'Zone Opted': ['zone opted'],
                      'Ambulance Limit': ['ambulance charges',
                                          'ambulance limit',
                                          'limit of additional ambulance charges per person',
                                          'limit of additional ambulance charges'],
                      'Number of Employees': ['no. of employees / members covered',
                                              'number of employees covered',
                                              'no of employees covered'],
                      'Number of Persons Covered': ['no. of persons covered',
                                                    'number of persons covered',
                                                    'total persons covered',
                                                    'no of persons covered'],
                      'Additional Cover Opted': ['additional cover opted', 'additional coverage opted'],
                      'Corporate Buffer': ['corporate buffer', 'corporate buffer limit'],
                      'Installment Facility': ['installment facility optn',
                                               'installment facility option',
                                               'instalment facility option'],
                      'Premium Frequency': ['premium payment frequency', 'frequency of premium payment'],
                      'Installment Amount': ['installment amount rs',
                                             'installment amount',
                                             'instalment amount'],
                      'Bonus': ['bonus', 'cumulative bonus', 'no claim bonus amount'],
                      'Limit of Coverage': ['limit of coverage', 'coverage limit'],
                      'Recharge Benefit': ['recharge benefit', 'recharge amount'],
                      'Family Size': ['scheme description (family size)',
                                      'scheme description family size',
                                      'family size'],
                      'Voluntary Co-payment': ['voluntary co-payment', 'voluntary co payment'],
                      'TPA Fax': ['tpa fax', 'tpa fax no'],
                      'Overall Policy Limit': ['overall policy limit'],
                      'Limit per Family': ['limit per family', 'family limit'],
                      'Cover Name': ['name of cover', 'cover name'],
                      'Nominee': ['nominee name',
                                  'name of nominee',
                                  'nominee',
                                  'nominee owner driver',
                                  'name of the nominee'],
                      'Nominee Relationship': ['nominee relationship',
                                               'relationship of nominee',
                                               'nominee relation',
                                               'relationship with nominee'],
                      'Policy Term': ['policy term', 'term of policy', 'policy duration'],
                      'Broker Name': ['broker name', 'name of broker', 'insurance broker name'],
                      'Broker Code': ['broker code', 'broker id', 'insurance broker code']},
 'Life Insurance': {'Life Assured': ['name of life assured', 'life assured name', 'life assured'],
                    'Nominee': ['name of nominee', 'nominee name', 'nominee'],
                    'Death Benefit': ['death benefit', 'sum assured on death'],
                    'Premium Frequency': ['frequency of premium payment',
                                          'premium payment frequency',
                                          'premium payment mode',
                                          'premium mode',
                                          'premium frequency',
                                          'frequency of payment',
                                          'mode of premium payment',
                                          'premium paying frequency'],
                    'Number of Members': ['number of members covered as on the date of commencement of '
                                          'master policy',
                                          'number of members covered',
                                          'number of members',
                                          'total eligible members',
                                          'total members covered',
                                          'number of lives covered',
                                          'total insured members',
                                          'no of members covered as on the date of commencement of '
                                          'master policy',
                                          'no of members covered',
                                          'no of members',
                                          'no of lives covered',
                                          'number of members covered as on the dt of commencement of '
                                          'master policy'],
                    'Maturity Date': ['maturity date',
                                      'date of maturity',
                                      'policy maturity date',
                                      'maturity dt',
                                      'dt of maturity',
                                      'policy maturity dt'],
                    'Appointee': ['appointee name', 'name of appointee', 'appointee'],
                    'Policy Term': ['policy term', 'term of policy', 'policy duration'],
                    'Premium Paying Term': ['premium paying term',
                                            'premium payment term',
                                            'premium payment duration'],
                    'Master Policy Number': ['master policy number',
                                             'master policy no',
                                             'master policy no.'],
                    'Scheme Name': ['name of the scheme', 'scheme name'],
                    'Group Type': ['type of group', 'group type'],
                    'Divisional Office': ['divisional office'],
                    'PGS Unit Office': ['p&gs unit office', 'pgs unit office'],
                    'Minimum Entry Age': ['minimum entry age', 'min entry age'],
                    'Maximum Entry Age': ['maximum entry age', 'max entry age'],
                    'Maximum Cover Ceasing Age': ['maximum cover ceasing age', 'cover ceasing age'],
                    'Premium Due Date': ['premium due date',
                                         'next premium due date',
                                         'premium due dt',
                                         'next premium due dt'],
                    'Rider Opted': ['rider opted', 'details of rider opted'],
                    'Rider Sum Assured': ['rider sum assured'],
                    'Nominee Relationship': ['nominee relationship',
                                             'relationship of nominee',
                                             'nominee relation',
                                             'relationship with nominee'],
                    'Policy Category': ['policy category', 'category of policy'],
                    'Client Type': ['client type', 'customer type'],
                    'First Policy Inception Date': ['date of inception of first policy',
                                                    'first policy inception date',
                                                    'dt of inception of first policy',
                                                    'first policy inception dt'],
                    'Agent Name': ['agent name', 'name of agent', 'insurance agent name'],
                    'Agent Code': ['agent code', 'agent id', 'insurance agent code']},
 'Others': {'Number of Members': ['number of members covered',
                                  'total members covered',
                                  'no of members covered'],
            'Accident Cover': ['personal accident cover', 'accident cover'],
            'Number of Students': ['no of students', 'no. of students', 'number of students'],
            'Number of Parents': ['no of parents', 'number of parents'],
            'Number of Lives': ['no.of lives', 'no of lives', 'number of lives'],
            'Medical Expenses per Student': ['medical expenses per student (inclusive of opd)',
                                             'medical expenses per student'],
            'Limit per Student': ['limit per student'],
            'Limit per Accident': ['limit per accident'],
            'Parents Sum Insured': ['total si of parents or guardian for payment of tuition and hostel '
                                    'fees'],
            'Tuition and Hostel Fees': ['payment of tuition and hostel fee for remaining semesters'],
            'Risk Group': ['risk group', 'risk class'],
            'Accidental Death Cover': ['accidental death cover', 'accidental death benefit'],
            'Permanent Total Disablement': ['permanent total disablement', 'permanent total disability'],
            'Permanent Partial Disablement': ['permanent partial disablement',
                                              'permanent partial disability'],
            'Cover Name': ['cover name', 'name of cover'],
            'Cover Definition': ['cover definition'],
            'Sum Insured Basis': ['sum insured (basis)', 'sum insured basis'],
            'Cremation Charges': ['cremation charges'],
            'Repatriation Benefit': ['repatriation of mortal remains', 'repatriation of remains'],
            'Client Type': ['client type', 'customer type'],
            'Co-insurance Share': ['own share', 'co insurance share'],
            'Scope of Cover': ['scope of cover'],
            'Risk Covered': ['risk covered'],
            'Special Exclusions': ['special exclusions'],
            'Special Perils': ['special perils'],
            'Broker Name': ['broker name', 'name of broker', 'insurance broker name'],
            'Broker Code': ['broker code', 'broker id', 'insurance broker code']}}
ALL_ALIASES = sorted(set(ALL_ALIASES).union(*(set(v) for s in [COMMON, PROPERTY_FIELDS, *SPECIFIC.values()] for v in s.values())), key=len, reverse=True)


# Final active name schema. Keep distinct legal roles only where needed.
_policyholder_labels = COMMON.pop("Policyholder", [])
_proposer_labels = COMMON.pop("Proposer Name", [])
COMMON.pop("Insured Person", None)
COMMON["Insured Name"] = list(dict.fromkeys(COMMON["Insured Name"] + ["name and address of insured", "name & address of insured", "insured name and address", "name of insured person", "insured person name", "name of the insured person"]))
for _category, _schema in SPECIFIC.items():
    for _field in ("Insured Person", "Policyholder", "Proposer Name"):
        _schema.pop(_field, None)
SPECIFIC["Life Insurance"]["Policyholder"] = _policyholder_labels
SPECIFIC["Life Insurance"]["Proposer Name"] = _proposer_labels
SPECIFIC["Family Insurance"]["Policyholder"] = _policyholder_labels
CUSTOMER_FIELDS.add("Insured Name")
ALL_ALIASES = sorted(set(ALL_ALIASES).union(*(set(v) for s in [COMMON, PROPERTY_FIELDS, *SPECIFIC.values()] for v in s.values())), key=len, reverse=True)
_NAME_FIELDS = {"Insured Name", "Policyholder", "Proposer Name", "Insured Person", "Life Assured", "Primary Insured", "Family Member", "Nominee", "Customer Name"}
_previous_value_from = _value_from
def _value_from(raw, field):
    if field in _NAME_FIELDS:
        raw = re.split(r"\b(?:period of insurance|policy period|policy number|policy no|date of birth|customer id|office code|issuing office|mobile number|email id)\b", _clean(raw), maxsplit=1, flags=re.I)[0].strip(" :-,|")
        return raw if validate_field(field, raw) else ""
    return _previous_value_from(raw, field)
