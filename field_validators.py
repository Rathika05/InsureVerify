"""
Insurance field validation.
Supports Vehicle, Health, Family, Life
and Other insurance categories.
Validates field values before they are
accepted by the extraction engine.
Source evidence and customer ownership
must be verified by the extraction engine.
"""
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
# ==========================================
# 1. COMMON TEXT CLEANING
# ==========================================
def clean_value(value):
    if value is None:
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip(" \t:|,;")
def is_missing(value):
    return clean_value(value).lower() in {
        "",
        "-",
        "--",
        "—",
        "–",
        "na",
        "n/a",
        "not found",
        "not available",
        "unknown"
    }
# ==========================================
# 2. DATE VALIDATION
# ==========================================
DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d %b, %Y",
    "%d %B, %Y"
)
TIME_FORMATS = (
    "",
    " %H:%M",
    " %H:%M:%S",
    " %I:%M %p",
    " %I:%M:%S %p",
    " %H:%M hrs",
    " %H:%M:%S hrs",
    "T%H:%M:%S"
)
def parse_date(value):
    value = clean_value(value)
    value = re.sub(
        r"(?i)(\d)(?:st|nd|rd|th)\b",
        r"\1",
        value
    )
    for date_format in DATE_FORMATS:
        for time_format in TIME_FORMATS:
            try:
                return datetime.strptime(
                    value,
                    date_format + time_format
                )
            except ValueError:
                continue
    return None
def valid_date(value):
    return parse_date(value) is not None
# ==========================================
# 3. MONEY VALIDATION
# ==========================================
def valid_money(value):
    value = clean_value(value)
    value = re.sub(
        r"^(?:₹|Rs\.?|INR)\s*",
        "",
        value,
        flags=re.I
    )
    value = re.sub(
        r"\s*(?:INR|rupees|/-)$",
        "",
        value,
        flags=re.I
    ).strip()
    pattern = (
        r"(?:"
        r"\d+|"
        r"\d{1,3}(?:,\d{3})+|"
        r"\d{1,2}(?:,\d{2})*,\d{3}"
        r")(?:\.\d{1,2})?"
    )
    if not re.fullmatch(pattern, value):
        return False
    try:
        amount = Decimal(
            value.replace(",", "")
        )
        return amount >= 0
    except InvalidOperation:
        return False
# ==========================================
# 4. POLICY NUMBER VALIDATION
# ==========================================
def valid_policy_number(value):
    value = clean_value(value)
    if is_missing(value):
        return False
    if valid_date(value):
        return False
    if re.search(
        r"\b(?:issuing|office|address|"
        r"policy|number|premium|mobile|"
        r"contact|registration|customer)\b",
        value,
        re.I
    ):
        return False
    # Support spaces around separators.
    value = re.sub(
        r"\s*([/-])\s*",
        r"\1",
        value
    )
    if not 5 <= len(value) <= 60:
        return False
    if not re.fullmatch(
        r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*",
        value
    ):
        return False
    if not re.search(r"\d", value):
        return False
    return True
# ==========================================
# 5. PERSON NAME VALIDATION
# ==========================================
def valid_person_name(value):
    value = clean_value(value)
    if is_missing(value):
        return False
    if not 2 <= len(value) <= 150:
        return False
    if not any(c.isalpha() for c in value):
        return False
    if any(c.isdigit() for c in value):
        return False
    if len(value.split()) > 18:
        return False
    invalid_words = (
        r"\b(?:phone|mobile|email|"
        r"rto|issuing|premium|"
        r"policy number|registration|"
        r"address|gender|date of birth|"
        r"customer id|authorised signatory|"
        r"authorized signatory|"
        r"terms and conditions)\b"
    )
    if re.search(
        invalid_words,
        value,
        re.I
    ):
        return False
    if value.lower() in {
        "name",
        "insured",
        "proposer",
        "policyholder",
        "not applicable",
        "none",
        "nil",
        "male",
        "female",
        "nominee",
        "doctor"
    }:
        return False
    return all(
        c.isalpha() or c in " .,'’-"
        for c in value
    )
# ==========================================
# 6. POLICYHOLDER / ORGANIZATION VALIDATION
# ==========================================
def valid_policyholder(value):
    value = clean_value(value)
    if valid_person_name(value):
        return True
    if is_missing(value):
        return False
    if not 2 <= len(value) <= 200:
        return False
    if not any(c.isalpha() for c in value):
        return False
    if re.search(
        r"\b(?:phone|mobile|email|"
        r"policy number|premium|"
        r"date of birth|customer id|"
        r"issuing office|servicing office|"
        r"registered office|"
        r"authorised signatory|"
        r"authorized signatory|"
        r"address)\b",
        value,
        re.I
    ):
        return False
    # Avoid accepting a full address as a name.
    if re.search(r"\b[1-9]\d{5}\b", value):
        return False
    organization_pattern = (
        r"\b(?:limited|ltd|pvt|"
        r"private|company|corporation|"
        r"institute|institution|"
        r"college|university|school|"
        r"association|foundation|"
        r"trust|society|bank|"
        r"industries|enterprises|"
        r"llp|inc|hospital|"
        r"technologies|technology|"
        r"services|solutions|"
        r"engineering|traders|"
        r"partners|firm|"
        r"cooperative|"
        r"co-operative)\b"
    )
    organization = bool(
        re.search(
            organization_pattern,
            value,
            re.I
        )
    )
    business_prefix = bool(
        re.match(
            r"^(?:M/s\.?|Messrs)\s*",
            value,
            re.I
        )
    )
    if any(c.isdigit() for c in value):
        if not organization and not business_prefix:
            return False
    if len(value.split()) > 25:
        return False
    return all(
        c.isalnum() or c in " .,&()'’-/+"
        for c in value
    )
# ==========================================
# 7. INSURER VALIDATION
# ==========================================
def valid_insurer(value):
    value = clean_value(value)
    if is_missing(value):
        return False
    if not 3 <= len(value) <= 200:
        return False
    if re.search(
        r"\b(?:policy number|"
        r"customer care|"
        r"authorised signatory|"
        r"registered office|"
        r"servicing office)\b",
        value,
        re.I
    ):
        return False
    return bool(
        re.search(
            r"\b(?:insurance|assurance|"
            r"insurer|LIC)\b",
            value,
            re.I
        )
    )
# ==========================================
# 8. MOBILE NUMBER VALIDATION
# ==========================================
def valid_mobile_number(value):
    value = clean_value(value)
    if not re.fullmatch(
        r"[+\d\s()-]+",
        value
    ):
        return False
    digits = re.sub(r"\D", "", value)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    elif value.startswith("+"):
        return False
    if not re.fullmatch(
        r"[6-9]\d{9}",
        digits
    ):
        return False
    if len(set(digits)) == 1:
        return False
    return True
# ==========================================
# 9. PHONE / LANDLINE VALIDATION
# ==========================================
def valid_phone_number(value):
    value = clean_value(value)
    if valid_mobile_number(value):
        return True
    if not re.fullmatch(
        r"[+\d\s()-]+",
        value
    ):
        return False
    digits = re.sub(r"\D", "", value)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    elif value.startswith("+"):
        return False
    if len(digits) != 10:
        return False
    if len(set(digits)) == 1:
        return False
    # Indian fixed-line area code + number.
    return bool(
        re.fullmatch(
            r"[1-8]\d{9}",
            digits
        )
    )
# ==========================================
# 10. EMAIL VALIDATION
# ==========================================
def valid_email(value):
    value = clean_value(value)
    if len(value) > 254:
        return False
    pattern = (
        r"[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9]"
        r"(?:[A-Za-z0-9-]*[A-Za-z0-9])?"
        r"(?:\.[A-Za-z0-9]"
        r"(?:[A-Za-z0-9-]*[A-Za-z0-9])?)*"
        r"\.[A-Za-z]{2,63}"
    )
    if not re.fullmatch(pattern, value):
        return False
    local, domain = value.rsplit("@", 1)
    if len(local) > 64:
        return False
    if local.startswith(".") or local.endswith("."):
        return False
    if ".." in value:
        return False
    if any(
        len(part) > 63
        for part in domain.split(".")
    ):
        return False
    return True
# ==========================================
# 11. CUSTOMER ID VALIDATION
# ==========================================
def valid_customer_id(value):
    value = clean_value(value)
    if not 3 <= len(value) <= 60:
        return False
    if valid_date(value):
        return False
    if not re.search(r"\d", value):
        return False
    return bool(
        re.fullmatch(
            r"[A-Za-z0-9/-]+",
            value
        )
    )
# ==========================================
# 12. GENDER VALIDATION
# ==========================================
def valid_gender(value):
    value = clean_value(value).lower()
    return value in {
        "male",
        "female",
        "other",
        "others",
        "transgender",
        "m",
        "f"
    }
# ==========================================
# 13. AGE VALIDATION
# ==========================================
def valid_age(value):
    value = clean_value(value)
    match = re.fullmatch(
        r"(\d{1,3})(?:\s*(?:years?|yrs?))?",
        value,
        re.I
    )
    if not match:
        return False
    return 0 <= int(match.group(1)) <= 120
# ==========================================
# 14. PINCODE VALIDATION
# ==========================================
def valid_pincode(value):
    value = clean_value(value)
    return bool(
        re.fullmatch(
            r"[1-9]\d{5}",
            value
        )
    )
# ==========================================
# 15. ADDRESS VALIDATION
# ==========================================
def valid_address(value):
    value = clean_value(value)
    if is_missing(value):
        return False
    if not 5 <= len(value) <= 700:
        return False
    if not any(c.isalpha() for c in value):
        return False
    if valid_email(value):
        return False
    if valid_phone_number(value):
        return False
    if re.search(
        r"\b(?:customer care|"
        r"toll free|helpline|"
        r"registered office|"
        r"corporate office|"
        r"issuing office|"
        r"policy servicing office|"
        r"agent code|broker code|"
        r"intermediary code|"
        r"grievance officer)\b",
        value,
        re.I
    ):
        return False
    if re.search(
        r"\b(?:policy number|"
        r"premium amount|"
        r"sum insured|"
        r"period of insurance|"
        r"date of birth)\b",
        value,
        re.I
    ):
        return False
    if re.search(
        r"(?i)^(?:means|shall|"
        r"terms and conditions|"
        r"provided that)\b",
        value
    ):
        return False
    return True
# ==========================================
# 16. CO-PAYMENT VALIDATION
# ==========================================
def valid_copayment(value):
    value = clean_value(value)
    if re.search(
        r"\b(?:phone|mobile|email|"
        r"address|telephone|contact)\b",
        value,
        re.I
    ):
        return False
    if value.lower() in {
        "nil",
        "none",
        "no",
        "no co-payment",
        "no copayment",
        "not applicable",
        "0",
        "zero"
    }:
        return True
    match = re.fullmatch(
        r"(\d{1,3}(?:\.\d+)?)\s*"
        r"(?:%|percent|per cent)",
        value,
        re.I
    )
    if match:
        percentage = float(match.group(1))
        return 0 <= percentage <= 100
    return False
# ==========================================
# 17. WAITING PERIOD VALIDATION
# ==========================================
def valid_waiting_period(value):
    value = clean_value(value)
    if value.lower() in {
        "nil",
        "none",
        "no waiting period",
        "not applicable",
        "zero",
        "0"
    }:
        return True
    return bool(
        re.fullmatch(
            r"\d{1,4}\s*"
            r"(?:days?|months?|years?|"
            r"weeks?|hrs?|hours?)",
            value,
            re.I
        )
    )
# ==========================================
# 18. VEHICLE REGISTRATION VALIDATION
# ==========================================
def valid_registration_number(value):
    value = clean_value(value)
    standard = re.fullmatch(
        r"[A-Z]{2}[ -]?\d{1,2}"
        r"[ -]?[A-Z]{0,3}[ -]?\d{1,4}",
        value,
        re.I
    )
    bharat_series = re.fullmatch(
        r"\d{2}[ -]?BH[ -]?"
        r"\d{4}[ -]?[A-Z]{1,2}",
        value,
        re.I
    )
    return bool(standard or bharat_series)
# ==========================================
# 19. ENGINE NUMBER VALIDATION
# ==========================================
def valid_engine_number(value):
    value = clean_value(value)
    return (
        bool(
            re.fullmatch(
                r"[A-Za-z0-9/-]{5,40}",
                value
            )
        )
        and bool(re.search(r"\d", value))
        and not valid_date(value)
    )
# ==========================================
# 20. CHASSIS NUMBER VALIDATION
# ==========================================
def valid_chassis_number(value):
    value = clean_value(value)
    return (
        bool(
            re.fullmatch(
                r"[A-Za-z0-9/-]{6,40}",
                value
            )
        )
        and bool(re.search(r"\d", value))
        and not valid_date(value)
    )
# ==========================================
# 21. CUBIC CAPACITY VALIDATION
# ==========================================
def valid_cc(value):
    value = clean_value(value)
    match = re.fullmatch(
        r"(\d{1,5})(?:\s*cc)?",
        value,
        re.I
    )
    return bool(
        match
        and 1 <= int(match.group(1)) <= 99999
    )
# ==========================================
# 22. MANUFACTURING YEAR VALIDATION
# ==========================================
def valid_manufacturing_year(value):
    value = clean_value(value)
    return (
        bool(re.fullmatch(r"\d{4}", value))
        and 1900 <= int(value) <= datetime.now().year + 1
    )
# ==========================================
# 23. SEATING CAPACITY VALIDATION
# ==========================================
def valid_seating_capacity(value):
    value = clean_value(value)
    match = re.fullmatch(
        r"(\d{1,3})(?:\s*seats?)?",
        value,
        re.I
    )
    return bool(
        match
        and 1 <= int(match.group(1)) <= 200
    )
# ==========================================
# 24. NUMBER OF MEMBERS VALIDATION
# ==========================================
def valid_member_count(value):
    value = clean_value(value)
    if not re.fullmatch(
        r"\d+|\d{1,3}(?:,\d{3})+|"
        r"\d{1,2}(?:,\d{2})*,\d{3}",
        value
    ):
        return False
    return int(value.replace(",", "")) > 0
# ==========================================
# 25. PREMIUM FREQUENCY VALIDATION
# ==========================================
def valid_premium_frequency(value):
    value = clean_value(value)
    return bool(
        re.fullmatch(
            r"annual|annually|yearly|"
            r"monthly|quarterly|"
            r"half[ -]?yearly|"
            r"semi[ -]?annual(?:ly)?|"
            r"single(?: premium)?|"
            r"one[ -]?time",
            value,
            re.I
        )
    )
# ==========================================
# 26. GENERAL TEXT VALIDATION
# ==========================================
def valid_general_text(value):
    value = clean_value(value)
    if is_missing(value):
        return False
    if len(value) > 700:
        return False
    if not any(c.isalnum() for c in value):
        return False
    if re.search(
        r"\b(?:following documents|"
        r"policy issuing|"
        r"how do i|"
        r"above changes|"
        r"authorised signatory|"
        r"authorized signatory)\b",
        value,
        re.I
    ):
        return False
    if re.match(
        r"^(?:means|shall|refers to|"
        r"must|should|will|"
        r"provided that)\b",
        value,
        re.I
    ):
        return False
    return True
# ==========================================
# 27. MAIN FIELD VALIDATOR
# ==========================================
def validate_field(field, value):
    field = clean_value(field).casefold()
    value = clean_value(value)
    if is_missing(value):
        return False
    # Date fields
    if field in {
        "policy start",
        "policy end",
        "renewal date",
        "date of birth",
        "policy start date",
        "policy end date",
        "expiry date",
        "commencement date",
        "issue date",
        "dob"
    }:
        return valid_date(value)
    # Policy number
    if field in {
        "policy number",
        "policy no",
        "master policy number",
        "certificate number"
    }:
        return valid_policy_number(value)
    # Insurance company
    if field in {
        "insurer",
        "insurance company",
        "insurer name"
    }:
        return valid_insurer(value)
    # Individual or organization
    if field in {
        "policyholder",
        "policyholder name",
        "policy holder name",
        "proposer name",
        "primary insured",
        "customer name",
        "insured name"
    }:
        return valid_policyholder(value)
    # Individual names
    if field in {
        "insured person",
        "insured person name",
        "family member",
        "life assured",
        "nominee",
        "doctor"
    }:
        return valid_person_name(value)
    # Mobile
    if field in {
        "mobile",
        "mobile number",
        "mobile no",
        "customer mobile",
        "customer mobile number"
    }:
        return valid_mobile_number(value)
    # Phone / landline
    if field in {
        "phone",
        "phone number",
        "customer phone",
        "contact number",
        "telephone",
        "telephone number",
        "landline",
        "landline number"
    }:
        return valid_phone_number(value)
    # Email
    if field in {
        "email id",
        "email",
        "email address",
        "customer email"
    }:
        return valid_email(value)
    # Customer ID
    if field in {
        "customer id",
        "client id",
        "member id",
        "insured id"
    }:
        return valid_customer_id(value)
    # Gender
    if field == "gender":
        return valid_gender(value)
    # Age
    if field == "age":
        return valid_age(value)
    # Pincode
    if field in {
        "pincode",
        "pin code",
        "postal code"
    }:
        return valid_pincode(value)
    # Address
    if field in {
        "residential address",
        "customer address",
        "insured address",
        "policyholder address",
        "address",
        "risk address",
        "location of risk"
    }:
        return valid_address(value)
    # Premium, taxes and coverage
    if field in {
        "premium",
        "annual premium",
        "gross premium",
        "net premium",
        "basic premium",
        "gst",
        "cgst",
        "sgst",
        "igst",
        "tax amount",
        "total premium",
        "total payable",
        "sum insured / assured",
        "sum insured",
        "sum assured",
        "total sum insured",
        "idv",
        "family floater cover",
        "death benefit",
        "coverage amount",
        "terrorism premium",
        "gvw"
    }:
        return valid_money(value)
    # Health insurance
    if field == "co-payment":
        return valid_copayment(value)
    if field == "waiting period":
        return valid_waiting_period(value)
    # Vehicle insurance
    if field == "registration number":
        return valid_registration_number(value)
    if field == "engine number":
        return valid_engine_number(value)
    if field == "chassis number":
        return valid_chassis_number(value)
    if field in {
        "cc",
        "cubic capacity",
        "engine capacity"
    }:
        return valid_cc(value)
    if field in {
        "manufacturing year",
        "registration year"
    }:
        return valid_manufacturing_year(value)
    if field in {
        "seating capacity",
        "seats"
    }:
        return valid_seating_capacity(value)
    # Group insurance
    if field in {
        "number of members",
        "total members",
        "insured members"
    }:
        return valid_member_count(value)
    # Life insurance
    if field == "premium frequency":
        return valid_premium_frequency(value)
    # Policy conditions
    if field in {
        "9 months waiting period deleted",
        "30 days waiting period deleted",
        "2/4 year exclusion deleted",
        "pre-existing cover opted"
    }:
        return value.upper() in ("YES", "NO")
    # Remaining fields
    return valid_general_text(value)


# Reject form labels and signature placeholders as names.
INVALID_NAME_PATTERN = re.compile(r"\b(?:company seal|office seal|rubber stamp|signature|authori[sz]ed signatory|period of insurance|policy period|policy number|policy no|date of birth|customer id|office code|terms and conditions|for office use only|declaration)\b", re.I)
_original_person_name = valid_person_name
_original_policyholder_name = valid_policyholder
def valid_person_name(value):
    return not INVALID_NAME_PATTERN.search(clean_value(value)) and _original_person_name(value)
def valid_policyholder(value):
    return not INVALID_NAME_PATTERN.search(clean_value(value)) and _original_policyholder_name(value)
