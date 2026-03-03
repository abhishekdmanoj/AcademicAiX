import os
import json
import shutil
import hashlib
import re
import requests
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "university_metadata.json")
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
ALLOWED_UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "data", "uploads")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

# ─────────────────────────────────────────
# EXAM RULES
# ─────────────────────────────────────────
EXAM_RULES = {
    "M.Tech": [{"name": "GATE", "website": "https://gate.iitk.ac.in"}],
    "M.E": [{"name": "GATE", "website": "https://gate.iitk.ac.in"}],
    "M.Sc": [{"name": "GATE", "website": "https://gate.iitk.ac.in"}],
    "M.Phil": [{"name": "CUET-PG", "website": "https://cuet.nta.nic.in"}],
    "MBA": [
        {"name": "CAT", "website": "https://iimcat.ac.in"},
        {"name": "CMAT", "website": "https://nta.ac.in/CMAT"},
        {"name": "KMAT", "website": "https://cee.kerala.gov.in"}
    ],
    "PGDM": [
        {"name": "CAT", "website": "https://iimcat.ac.in"},
        {"name": "CMAT", "website": "https://nta.ac.in/CMAT"}
    ],
    "MCA": [{"name": "NIMCET", "website": "https://nimcet.admissions.nic.in"}],
    "M.Com": [{"name": "CUET-PG", "website": "https://cuet.nta.nic.in"}],
    "M.A": [{"name": "CUET-PG", "website": "https://cuet.nta.nic.in"}],
    "MA": [{"name": "CUET-PG", "website": "https://cuet.nta.nic.in"}],
    "MSW": [{"name": "CUET-PG", "website": "https://cuet.nta.nic.in"}],
    "M.Ed": [{"name": "CUET-PG", "website": "https://cuet.nta.nic.in"}],
    "LLM": [{"name": "CLAT-PG", "website": "https://consortiumofnlus.ac.in"}],
    "B.Tech": [{"name": "JEE Main", "website": "https://jeemain.nta.nic.in"}],
    "B.E": [{"name": "JEE Main", "website": "https://jeemain.nta.nic.in"}],
    "B.Arch": [
        {"name": "JEE Main", "website": "https://jeemain.nta.nic.in"},
        {"name": "NATA", "website": "https://nata.in"}
    ],
    "MBBS": [{"name": "NEET-UG", "website": "https://neet.nta.nic.in"}],
    "BDS": [{"name": "NEET-UG", "website": "https://neet.nta.nic.in"}],
    "BAMS": [{"name": "NEET-UG", "website": "https://neet.nta.nic.in"}],
    "BHMS": [{"name": "NEET-UG", "website": "https://neet.nta.nic.in"}],
    "BPT": [{"name": "NEET-UG", "website": "https://neet.nta.nic.in"}],
    "B.Pharm": [{"name": "NEET-UG", "website": "https://neet.nta.nic.in"}],
    "B.Com": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
    "B.A": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
    "BA": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
    "B.Sc": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
    "BCA": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
    "BBA": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
    "BBM": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
    "B.Ed": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
    "BSW": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
    "LLB": [{"name": "CLAT", "website": "https://consortiumofnlus.ac.in"}],
    "B.Des": [{"name": "UCEED", "website": "https://www.iitb.ac.in/uceed"}],
    "BFA": [{"name": "CUET-UG", "website": "https://cuet.nta.nic.in"}],
}

PG_DEGREES = [
    "M.Tech", "M.E", "M.Sc", "M.Phil",
    "MBA", "PGDM", "MCA", "M.Com",
    "M.A", "MA", "MSW", "M.Ed", "LLM"
]

DEGREE_PATTERNS = [
    r"M\.Tech\.?\s+(?:in\s+)?([A-Za-z\s&\(\)]+)",
    r"M\.E\.?\s+(?:in\s+)?([A-Za-z\s&]+)",
    r"M\.Sc\.?\s+(?:in\s+)?([A-Za-z\s&]+)",
    r"M\.Phil\.?\s+(?:in\s+)?([A-Za-z\s&]+)",
    r"MBA\s*[-–]?\s*(?:in\s+)?([A-Za-z\s&]*)",
    r"PGDM\s*[-–]?\s*(?:in\s+)?([A-Za-z\s&]*)",
    r"MCA",
    r"M\.Com\.?\s+(?:in\s+)?([A-Za-z\s&]*)",
    r"M\.A\.?\s+(?:in\s+)?([A-Za-z\s&]+)",
    r"\bMA\b\s+(?:in\s+)?([A-Za-z\s&]+)",
    r"MSW", r"M\.Ed", r"LLM",
    r"B\.Tech\.?\s+(?:in\s+)?([A-Za-z\s&\(\)]+)",
    r"B\.E\.?\s+(?:in\s+)?([A-Za-z\s&]+)",
    r"B\.Arch",
    r"MBBS", r"BDS", r"BAMS", r"BHMS", r"BPT", r"B\.Pharm",
    r"B\.Com\.?\s*\(?(?:Hons\.?)?\)?",
    r"B\.A\.?\s*\(?(?:Hons\.?)?\)?\s*(?:in\s+)?([A-Za-z\s&]+)",
    r"\bBA\b\s*\(?(?:Hons\.?)?\)?\s*(?:in\s+)?([A-Za-z\s&]+)",
    r"B\.Sc\.?\s+(?:in\s+)?([A-Za-z\s&]+)",
    r"BCA", r"BBA", r"BBM", r"B\.Ed", r"BSW", r"LLB",
    r"B\.Des", r"BFA",
]

# ─────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────

def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def sanitize_filename(name):
    """Remove or replace characters unsafe for filenames"""
    return re.sub(r'[^A-Za-z0-9_\-]', '_', name).strip('_')


def is_ollama_running():
    try:
        r = requests.get("http://localhost:11434", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────
# FIX 1: VERSION-AWARE DUPLICATE HANDLING
# ─────────────────────────────────────────

def handle_existing_entry(college, program, registry):
    """
    If college + program already exists:
    - Check if hash changed (new version)
    - If same hash → skip (no change)
    - If different hash → deactivate old, allow new insert
    - If not found → allow fresh insert
    Returns: "skip" | "update" | "new"
    """
    for entry in registry:
        if (
            entry.get("college") == college and
            entry.get("program") == program and
            entry.get("is_active", False)
        ):
            return "update"
    return "new"


def deactivate_existing(college, program, registry):
    """Deactivate all active entries for this college+program"""
    for entry in registry:
        if (
            entry.get("college") == college and
            entry.get("program") == program and
            entry.get("is_active", False)
        ):
            entry["is_active"] = False
    return registry


# ─────────────────────────────────────────
# LAYER 1: REGEX EXTRACTION
# ─────────────────────────────────────────

def extract_text_from_pdf(pdf_path, max_pages=4):
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc[:max_pages]:
        text += page.get_text()
    return text


def extract_college_name(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    priority_keywords = [
        "national institute of technology",
        "indian institute of technology",
        "indian institute of management",
        "university",
        "college",
        "institute",
        "school of",
        "academy",
    ]
    for line in lines[:30]:
        line_lower = line.lower()
        if any(kw in line_lower for kw in priority_keywords):
            clean = re.sub(r'\s+', ' ', line).strip()
            if 5 < len(clean) < 100:
                return clean
    return None


def extract_program_name(text):
    clean_text = re.sub(r'\s+', ' ', text[:2000])
    garbage_words = [
        "curriculum", "syllabus", "scheme", "regulations",
        "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"
    ]
    for pattern in DEGREE_PATTERNS:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            full_match = match.group(0).strip()
            full_match = re.sub(r'\s+', ' ', full_match)
            full_match = full_match.rstrip('.,;:')
            for word in garbage_words:
                full_match = re.sub(
                    rf'\b{word}\b', '',
                    full_match,
                    flags=re.IGNORECASE
                ).strip()
            if len(full_match) > 2:
                return full_match
    return None


def infer_degree_level(program_name):
    if not program_name:
        return "UG"
    for deg in PG_DEGREES:
        if deg.lower() in program_name.lower():
            return "PG"
    return "UG"


def infer_entrance_exams(program_name):
    if not program_name:
        return []
    for key in EXAM_RULES:
        if key.lower() in program_name.lower():
            return EXAM_RULES[key]
    return []


# ─────────────────────────────────────────
# LAYER 2: OLLAMA FALLBACK
# ─────────────────────────────────────────

def extract_with_ollama(text):
    prompt = (
        "Return only a JSON object, nothing else. "
        "Extract the university name and academic program name. "
        "Rules: college is the institution name only, not a city. "
        "Program should contain only degree type and subject. "
        "Remove city names, years, and words like "
        "Curriculum, Syllabus, Scheme from program name. "
        f"Text: {text[:1500]} "
        "JSON: {\"college\": \"\", \"program\": \"\"}"
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30
        )
        raw = response.json()["response"].strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        parsed = json.loads(raw)
        college = parsed.get("college", "").strip() or None
        program = parsed.get("program", "").strip() or None
        if college and len(college) < 3:
            college = None
        if program and len(program) < 2:
            program = None
        return {"college": college, "program": program}
    except Exception as e:
        print(f"⚠️  Ollama extraction failed: {e}")
        return {"college": None, "program": None}


# ─────────────────────────────────────────
# COMBINED EXTRACTION
# ─────────────────────────────────────────

def extract_metadata(pdf_path):
    """
    Layer 1: Regex
    Layer 2: Ollama fallback for failed fields
    Layer 3: Admin correction handled by caller
    Does NOT embed. Does NOT touch FAISS.
    """
    text = extract_text_from_pdf(pdf_path)
    college = extract_college_name(text)
    program = extract_program_name(text)

    if not college or not program:
        if is_ollama_running():
            print("🤖 Regex incomplete. Trying Ollama fallback...")
            result = extract_with_ollama(text)
            college = college or result.get("college")
            program = program or result.get("program")
        else:
            print("⚠️  Ollama not running. Skipping LLM fallback.")

    return {
        "college": college or "",
        "program": program or "",
        "degree_level": infer_degree_level(program),
        "entrance_exams": infer_entrance_exams(program or "")
    }


# ─────────────────────────────────────────
# REGISTRY + METADATA UPDATE
# ─────────────────────────────────────────

def register_program(
    pdf_path,
    college,
    program,
    degree_level,
    country="India",
    state="",
    source_url=""
):
    """
    Registers program into both JSON files.
    Version-aware: deactivates old entry if exists.
    Does NOT embed. Does NOT rebuild FAISS.
    Caller triggers rebuild.
    """
    registry = load_json(REGISTRY_PATH)
    university_metadata = load_json(METADATA_PATH)

    # FIX 1: Version-aware handling
    status = handle_existing_entry(college, program, registry)

    if status == "update":
        print(f"🔄 Existing entry found. Deactivating old version...")
        registry = deactivate_existing(college, program, registry)
    
    # FIX 2: Sanitize filename
    safe_college = sanitize_filename(college)
    safe_program = sanitize_filename(program)
    clean_name = f"{safe_college}_{safe_program}.pdf"

    os.makedirs(PDF_DIR, exist_ok=True)
    dest_path = os.path.join(PDF_DIR, clean_name)

    if os.path.abspath(pdf_path) != os.path.abspath(dest_path):
        shutil.copy2(pdf_path, dest_path)

    relative_path = f"data/raw_pdfs/{clean_name}"
    file_hash = compute_sha256(dest_path)
    entrance_exams = infer_entrance_exams(program)

    registry.append({
        "college": college,
        "program": program,
        "degree_level": degree_level,
        "file_path": relative_path,
        "source_url": source_url,
        "hash": file_hash,
        "academic_year": "",
        "country": country,
        "state": state,
        "is_active": True,
        "last_checked": str(datetime.now().date())
    })

    # Update university_metadata
    # Deactivate old metadata entry if exists
    university_metadata = [
        m for m in university_metadata
        if not (
            m.get("college") == college and
            m.get("program") == program
        )
    ]

    university_metadata.append({
        "college": college,
        "program": program,
        "official_website": "",
        "entrance_exams": entrance_exams,
        "pyq_links": [],
        "last_updated": str(datetime.now().date())
    })

    save_json(REGISTRY_PATH, registry)
    save_json(METADATA_PATH, university_metadata)

    action = "Updated" if status == "update" else "Registered"
    return {
        "success": True,
        "message": f"{action}: {college} - {program}"
    }


# ─────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────

def cli_ingest(pdf_path, country="India", state="", source_url=""):
    print(f"\n📄 Processing: {os.path.basename(pdf_path)}")

    try:
        metadata = extract_metadata(pdf_path)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

    print(f"\n🔍 Detected:")
    print(f"   College  : {metadata['college'] or 'NOT DETECTED'}")
    print(f"   Program  : {metadata['program'] or 'NOT DETECTED'}")
    print(f"   Degree   : {metadata['degree_level']}")
    print(f"   Exams    : {[e['name'] for e in metadata['entrance_exams']]}")

    print("\n📝 Press Enter to accept or type a correction:")
    college_input = input(
        f"   College [{metadata['college']}]: "
    ).strip()
    program_input = input(
        f"   Program [{metadata['program']}]: "
    ).strip()

    final_college = college_input or metadata["college"]
    final_program = program_input or metadata["program"]

    if not final_college or not final_program:
        print("❌ College and program required. Aborting.")
        return False

    print(f"\n✅ Final:")
    print(f"   College : {final_college}")
    print(f"   Program : {final_program}")
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("⚠️  Cancelled.")
        return False

    result = register_program(
        pdf_path=pdf_path,
        college=final_college,
        program=final_program,
        degree_level=infer_degree_level(final_program),
        country=country,
        state=state,
        source_url=source_url
    )

    print(f"✅ {result['message']}")
    return True


def cli_ingest_folder(folder_path, country="India", state=""):
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    pdfs = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdfs:
        print("❌ No PDFs found.")
        return

    print(f"🔍 Found {len(pdfs)} PDFs")
    success_count = 0

    for pdf_file in pdfs:
        result = cli_ingest(
            os.path.join(folder_path, pdf_file),
            country=country,
            state=state
        )
        if result:
            success_count += 1

    print(f"\n📊 {success_count}/{len(pdfs)} ingested.")

    # FIX 4: Rebuild ONCE after all ingestions
    if success_count > 0:
        print("\n🚀 Rebuilding index...")
        from offline_pipeline.build_syllabus_index import build_syllabus_index
        build_syllabus_index()
        print("✅ Index rebuilt. All programs live.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m ingestion.auto_ingest <pdf_path>")
        print("  python -m ingestion.auto_ingest <folder_path>")
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isdir(path):
        cli_ingest_folder(path)
    elif os.path.isfile(path) and path.endswith(".pdf"):
        success = cli_ingest(path)
        if success:
            print("\n🚀 Rebuilding index...")
            from offline_pipeline.build_syllabus_index import build_syllabus_index
            build_syllabus_index()
            print("✅ Done.")
    else:
        print("❌ Invalid path or not a PDF.")