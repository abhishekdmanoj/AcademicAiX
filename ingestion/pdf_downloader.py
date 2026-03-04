import os
import json
import hashlib
import requests
import re
from urllib.parse import urljoin, urlparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCES_PATH = os.path.join(PROJECT_ROOT, "data", "sources.json")
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "data", "registry.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# STRICT whitelist — filename must contain at least one of these
MUST_CONTAIN = [
    "syllabus", "curriculum", "scheme", "course-outline",
    "courseoutline", "course_outline", "programme-structure",
    "program-structure", "study-plan", "studyplan",
    "academic-plan", "module-guide", "moduleguide",
    "subject-outline", "unit-outline"
]

# Hard reject — if filename contains any of these, always skip
MUST_REJECT = [
    "fee", "timetable", "calendar", "hostel", "application",
    "admission", "prospectus", "brochure", "scholarship",
    "notice", "circular", "tender", "result", "admit",
    "hall ticket", "marksheet", "harassment", "yoga",
    "gender", "medal", "award", "orientation", "fellowship",
    "refund", "withdrawal", "verification", "regulation",
    "rules", "constitution", "dasa", "form", "report",
    "certificate", "cgpa", "multiplication", "web links",
    "document 1", "events", "pgp", "sfs"
]


# ─────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def is_already_downloaded(file_hash, registry):
    for entry in registry:
        if entry.get("hash") == file_hash:
            return True
    return False


def is_syllabus_filename(filename):
    """
    STRICT whitelist approach.
    Default = REJECT.
    Only accept if filename contains a syllabus keyword
    AND does not contain a reject keyword.
    """
    name = filename.lower().replace("%20", " ").replace("-", " ").replace("_", " ")

    # Hard reject first
    for word in MUST_REJECT:
        if word in name:
            return False

    # Only accept if explicitly looks like a syllabus
    for word in MUST_CONTAIN:
        if word.replace("-", " ") in name:
            return True

    # Default REJECT
    return False


# ─────────────────────────────────────────
# DOWNLOAD SINGLE PDF
# ─────────────────────────────────────────

def download_pdf(url, save_path, timeout=20):
    try:
        response = requests.get(
            url, headers=HEADERS, timeout=timeout,
            stream=True, verify=False
        )
        response.raise_for_status()

        content = b""
        for chunk in response.iter_content(8192):
            content += chunk

        if not content.startswith(b"%PDF"):
            print(f"   ⚠ Not a valid PDF: {os.path.basename(url)}")
            return False

        with open(save_path, "wb") as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False


# ─────────────────────────────────────────
# SCRAPE PAGE FOR PDF LINKS
# ─────────────────────────────────────────

def scrape_pdf_links(page_url, timeout=15):
    try:
        response = requests.get(
            page_url, headers=HEADERS,
            timeout=timeout, verify=False
        )
        response.raise_for_status()
        html = response.text

        pdf_pattern = re.compile(
            r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
            re.IGNORECASE
        )
        matches = pdf_pattern.findall(html)
        pdf_urls = list(set([urljoin(page_url, m) for m in matches]))
        print(f"   Found {len(pdf_urls)} PDF links on page")
        return pdf_urls

    except Exception as e:
        print(f"   ❌ Failed to scrape page: {e}")
        return []


# ─────────────────────────────────────────
# PROCESS ONE SOURCE
# ─────────────────────────────────────────

def process_source(source, registry):
    college = source["college"]
    country = source["country"]
    state = source.get("state", "")
    source_type = source.get("type", "page")
    urls = source.get("urls", [])

    print(f"\n🏫 {college} ({country})")

    downloaded = []
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for url in urls:
        is_direct = source_type == "direct_pdf" or url.lower().endswith(".pdf")

        if is_direct:
            filename = os.path.basename(urlparse(url).path)
            if not filename.endswith(".pdf"):
                filename = f"{college.replace(' ', '_')}.pdf"

            save_path = os.path.join(DOWNLOAD_DIR, filename)
            print(f"   📥 Downloading: {filename}")

            success = download_pdf(url, save_path)
            if not success:
                continue

            file_hash = compute_sha256(save_path)
            if is_already_downloaded(file_hash, registry):
                print(f"   ✅ Already indexed, skipping")
                os.remove(save_path)
                continue

            downloaded.append({
                "file_path": save_path,
                "source_url": url,
                "college": college,
                "country": country,
                "state": state
            })
            print(f"   ✅ Saved")

        else:
            print(f"   🔍 Scraping: {url}")
            pdf_links = scrape_pdf_links(url)

            for pdf_url in pdf_links:
                filename = os.path.basename(urlparse(pdf_url).path)

                if not is_syllabus_filename(filename):
                    continue

                save_path = os.path.join(DOWNLOAD_DIR, filename)
                if os.path.exists(save_path):
                    continue

                print(f"   📥 {filename}")
                success = download_pdf(pdf_url, save_path)
                if not success:
                    continue

                file_hash = compute_sha256(save_path)
                if is_already_downloaded(file_hash, registry):
                    print(f"   ✅ Already indexed, skipping")
                    os.remove(save_path)
                    continue

                downloaded.append({
                    "file_path": save_path,
                    "source_url": pdf_url,
                    "college": college,
                    "country": country,
                    "state": state
                })
                print(f"   ✅ Saved")

    print(f"   📊 {len(downloaded)} new PDFs")
    return downloaded


# ─────────────────────────────────────────
# MAIN BULK DOWNLOAD
# ─────────────────────────────────────────

def bulk_download(sources_path=None):
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if sources_path is None:
        sources_path = SOURCES_PATH

    if not os.path.exists(sources_path):
        print(f"❌ sources.json not found at {sources_path}")
        return {"success": False, "message": "sources.json not found"}

    sources = load_json(sources_path)
    registry = load_json(REGISTRY_PATH)

    print(f"🚀 Bulk download from {len(sources)} sources...")
    print(f"📋 Registry has {len(registry)} entries\n")

    all_downloaded = []

    for source in sources:
        try:
            downloaded = process_source(source, registry)
            all_downloaded.extend(downloaded)
        except Exception as e:
            print(f"❌ Error: {source.get('college')}: {e}")

    print(f"\n📦 Total new PDFs: {len(all_downloaded)}")

    if not all_downloaded:
        print("ℹ Nothing new to ingest.")
        return {"success": True, "ingested": 0}

    print("\n🔄 Running auto_ingest...")

    from ingestion.auto_ingest import (
        extract_metadata, register_program, infer_degree_level
    )

    success_count = 0

    for item in all_downloaded:
        pdf_path = item["file_path"]
        college = item["college"]
        country = item["country"]
        state = item["state"]
        source_url = item["source_url"]

        print(f"\n📄 {os.path.basename(pdf_path)}")

        try:
            metadata = extract_metadata(pdf_path)

            program = metadata.get("program", "").strip()
            if not program or program.lower() in [
                "na", "n/a", "not provided",
                "degree type and subject", ""
            ]:
                raw_name = os.path.splitext(
                    os.path.basename(pdf_path)
                )[0]
                program = raw_name.replace("_", " ").replace("-", " ").strip()

            degree_level = metadata.get("degree_level", "UG")

            print(f"   College : {college}")
            print(f"   Program : {program}")
            print(f"   Degree  : {degree_level}")

            result = register_program(
                pdf_path=pdf_path,
                college=college,
                program=program,
                degree_level=degree_level,
                country=country,
                state=state,
                source_url=source_url
            )

            if result["success"]:
                print(f"   ✅ {result['message']}")
                success_count += 1
            else:
                print(f"   ⚠ {result['message']}")

        except Exception as e:
            print(f"   ❌ {e}")

    print(f"\n📊 Ingested: {success_count}/{len(all_downloaded)}")

    if success_count > 0:
        print("\n🚀 Rebuilding index...")
        try:
            from offline_pipeline.build_syllabus_index import build_syllabus_index
            build_syllabus_index()
            print("✅ Done. All programs live.")
        except Exception as e:
            print(f"❌ Rebuild failed: {e}")

    return {"success": True, "ingested": success_count}


if __name__ == "__main__":
    import sys
    sources_path = sys.argv[1] if len(sys.argv) > 1 else None
    bulk_download(sources_path)
