import os
import json
import hashlib
import requests
import re
import time
from urllib.parse import urljoin, urlparse, unquote

# Suppress SSL warnings globally — many Indian university sites have bad certs
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

# Anchor text hints for crawl mode (looser than filename filter)
ANCHOR_SYLLABUS_HINTS = [
    "syllabus", "curriculum", "scheme", "btech", "b.tech",
    "mtech", "m.tech", "regulations", "study plan",
    "course structure", "programme", "academic plan"
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

DEFAULT_MIN_YEAR = 2018


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
    name = filename.lower().replace("%20", " ").replace("-", " ").replace("_", " ")
    for word in MUST_REJECT:
        if word in name:
            return False
    for word in MUST_CONTAIN:
        if word.replace("-", " ") in name:
            return True
    return False


def is_relevant_crawl_pdf(pdf_url, anchor_text="", start_url=""):
    """
    Relevance check for crawled PDFs.
    If the crawl start URL is itself a syllabus page, trust the context
    and accept everything that isn't explicitly rejected.
    Otherwise require filename or anchor text to hint at syllabus content.
    """
    filename = unquote(os.path.basename(urlparse(pdf_url).path)).lower()
    anchor = anchor_text.lower()

    # Hard reject by filename always
    for word in MUST_REJECT:
        if word in filename:
            return False

    # If the start URL path contains "syllabus", we're already in the right
    # subtree — accept everything not explicitly rejected above
    start_path = urlparse(start_url).path.lower() if start_url else ""
    if "syllabus" in start_path or "curriculum" in start_path:
        return True

    # Otherwise require filename or anchor to suggest syllabus content
    for word in MUST_CONTAIN:
        if word.replace("-", " ") in filename.replace("-", " ").replace("_", " "):
            return True

    for hint in ANCHOR_SYLLABUS_HINTS:
        if hint in anchor:
            return True

    return False


# ─────────────────────────────────────────
# YEAR FILTER — 3 layers
# ─────────────────────────────────────────

def extract_year_from_string(text):
    """Find the most recent 4-digit year (2000–2035) in a string."""
    matches = re.findall(r'\b(20[0-2][0-9]|2035)\b', text)
    if matches:
        return max(int(y) for y in matches)
    return None


def get_pdf_year(pdf_url, pdf_path=None):
    """
    3-layer year detection:
    1. Year in URL/filename
    2. PDF metadata creation date
    3. Year mentioned in first 2 pages of content
    Returns (year, source) or (None, "unverified")
    """
    # Layer 1 — URL/filename
    url_year = extract_year_from_string(pdf_url)
    if url_year:
        return url_year, "url"

    if pdf_path and os.path.exists(pdf_path):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)

            # Layer 2 — PDF metadata
            creation = doc.metadata.get("creationDate", "")
            if creation:
                meta_year = extract_year_from_string(creation)
                if meta_year:
                    doc.close()
                    return meta_year, "metadata"

            # Layer 3 — content scan (first 2 pages)
            content_text = ""
            for page_num in range(min(2, len(doc))):
                content_text += doc[page_num].get_text()
            doc.close()

            content_year = extract_year_from_string(content_text)
            if content_year:
                return content_year, "content"

        except Exception:
            pass

    return None, "unverified"


def is_year_acceptable(pdf_url, pdf_path=None, min_year=DEFAULT_MIN_YEAR):
    """
    Returns (acceptable, year, source).
    Unverified year → accept with flag rather than silently reject.
    """
    year, source = get_pdf_year(pdf_url, pdf_path)

    if source == "unverified":
        return True, None, "unverified"

    if year and year >= min_year:
        return True, year, source

    return False, year, source


# ─────────────────────────────────────────
# GOOGLE SEARCH — auto find start URL
# ─────────────────────────────────────────

def google_find_start_url(college, domain):
    """
    Search Google for the best syllabus page on a university's domain.
    Called automatically when start_url is blank in sources.json.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    cse_id  = os.environ.get("GOOGLE_CSE_ID", "")

    if not api_key or not cse_id:
        print(f"   ⚠ GOOGLE_API_KEY/CSE_ID not set — will crawl from domain root")
        return None

    netloc = urlparse(domain).netloc
    query  = f"{college} syllabus filetype:pdf site:{netloc}"

    try:
        res = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cse_id, "q": query, "num": 5},
            timeout=10,
            verify=False
        )
        items = res.json().get("items", [])

        # Prefer non-PDF listing pages
        for item in items:
            link = item.get("link", "")
            if not link.lower().endswith(".pdf") and netloc in urlparse(link).netloc:
                print(f"   🔍 Auto start URL found: {link}")
                return link

        # Fallback — use parent directory of first PDF result
        if items:
            first = items[0].get("link", "")
            parent = first.rsplit("/", 1)[0] + "/"
            print(f"   🔍 Auto start URL (parent dir): {parent}")
            return parent

    except Exception as e:
        print(f"   ⚠ Google search failed: {e}")

    return None


# ─────────────────────────────────────────
# RECURSIVE CRAWLER
# ─────────────────────────────────────────

def crawl_for_pdfs(start_url, domain, max_depth=3, crawl_delay=0.3):
    """
    Recursively crawl a university website for PDF links.
    Stays within the start URL subtree only — won't follow nav/footer links
    back to homepage or unrelated sections. Returns [(pdf_url, anchor_text), ...]
    """
    netloc     = urlparse(domain).netloc
    start_path = urlparse(start_url).path.rstrip("/")
    visited    = set()
    pdf_results = []

    def crawl_page(url, depth):
        if depth > max_depth or url in visited:
            return
        parsed = urlparse(url)
        if netloc not in parsed.netloc:
            return
        # Stay within the starting subtree (PDFs anywhere on domain are OK)
        if not url.lower().endswith(".pdf") and ".pdf?" not in url.lower():
            if not parsed.path.startswith(start_path):
                return

        visited.add(url)
        print(f"   🕷  Crawling (depth {depth}): {url}")

        html = None

        # Try requests first (fast) — Playwright only if requests returns no links
        try:
            res = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            res.raise_for_status()
            html = res.text
        except Exception as e:
            print(f"   ⚠ Could not fetch {url}: {e}")
            return

        if not html:
            return

        # Extract all <a href> links with anchor text
        link_pattern = re.compile(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL
        )
        skip_exts = (".jpg", ".jpeg", ".png", ".gif", ".css", ".js",
                     ".zip", ".rar", ".mp4", ".mp3", ".docx", ".xlsx")

        for match in link_pattern.finditer(html):
            href    = match.group(1).strip()
            anchor  = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            abs_url = urljoin(url, href)
            parsed  = urlparse(abs_url)

            if abs_url.lower().endswith(".pdf") or ".pdf?" in abs_url.lower():
                if abs_url not in [r[0] for r in pdf_results]:
                    pdf_results.append((abs_url, anchor))
                    print(f"   📎 PDF found: {os.path.basename(urlparse(abs_url).path)}")
            elif netloc in parsed.netloc:
                if not any(parsed.path.lower().endswith(e) for e in skip_exts):
                    time.sleep(crawl_delay)
                    crawl_page(abs_url, depth + 1)

    crawl_page(start_url, 0)
    print(f"   🗂  Crawl complete — {len(pdf_results)} PDF links found")
    return pdf_results


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
        matches  = pdf_pattern.findall(html)
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
    college     = source["college"]
    country     = source["country"]
    state       = source.get("state", "")
    source_type = source.get("type", "page")
    urls        = source.get("urls", [])
    use_filename_as_program = source.get("skip_filename_filter", False)

    print(f"\n🏫 {college} ({country})")

    downloaded = []
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # ── CRAWL ─────────────────────────────────────────────────────
    if source_type == "crawl":
        domain    = source.get("domain", "")
        start_url = source.get("start_url", "").strip()
        max_depth = source.get("max_depth", 3)
        min_year  = source.get("min_year", DEFAULT_MIN_YEAR)

        if not domain:
            print(f"   ❌ No domain set — skipping")
            return []

        # Auto-find start URL if not provided
        if not start_url:
            print(f"   🔍 No start URL — querying Google...")
            start_url = google_find_start_url(college, domain)
            if not start_url:
                start_url = domain
                print(f"   ⚠ Falling back to domain root: {start_url}")

        print(f"   🕷  Crawling from: {start_url} (depth {max_depth}, min year {min_year})")
        pdf_links = crawl_for_pdfs(start_url, domain, max_depth=max_depth)

        for pdf_url, anchor_text in pdf_links:
            filename = unquote(os.path.basename(urlparse(pdf_url).path))

            if not is_relevant_crawl_pdf(pdf_url, anchor_text, start_url):
                continue

            save_path = os.path.join(DOWNLOAD_DIR, filename)
            if os.path.exists(save_path):
                continue

            print(f"   📥 {filename}")
            success = download_pdf(pdf_url, save_path)
            if not success:
                continue

            # Year filter
            acceptable, year, year_source = is_year_acceptable(pdf_url, save_path, min_year)
            if not acceptable:
                print(f"   🗓  Rejected — year {year} < {min_year}")
                os.remove(save_path)
                continue

            if year_source == "unverified":
                print(f"   🗓  Year unverified — accepting with flag")
            else:
                print(f"   🗓  Year {year} via {year_source} ✅")

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
                "state": state,
                "use_filename_as_program": True,
                "year_verified": year_source != "unverified",
            })
            print(f"   ✅ Saved")

        print(f"   📊 {len(downloaded)} new PDFs from crawl")
        return downloaded

    # ── DIRECT PDF / PAGE ─────────────────────────────────────────
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
                "state": state,
                "use_filename_as_program": use_filename_as_program
            })
            print(f"   ✅ Saved")

        else:
            print(f"   🔍 Scraping: {url}")
            pdf_links = scrape_pdf_links(url)

            for pdf_url in pdf_links:
                filename = os.path.basename(urlparse(pdf_url).path)

                if source_type != "page" and not is_syllabus_filename(filename):
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
                    "state": state,
                    "use_filename_as_program": use_filename_as_program
                })
                print(f"   ✅ Saved")

    print(f"   📊 {len(downloaded)} new PDFs")
    return downloaded


# ─────────────────────────────────────────
# MAIN BULK DOWNLOAD
# ─────────────────────────────────────────

def bulk_download(sources_path=None):
    from dotenv import load_dotenv
    load_dotenv()

    if sources_path is None:
        sources_path = SOURCES_PATH

    if not os.path.exists(sources_path):
        print(f"❌ sources.json not found at {sources_path}")
        return {"success": False, "message": "sources.json not found"}

    sources  = load_json(sources_path)
    registry = load_json(REGISTRY_PATH)

    print(f"🚀 Bulk download from {len(sources)} sources...")
    print(f"📋 Registry has {len(registry)} entries\n")

    all_downloaded = []

    for source in sources:
        try:
            downloaded = process_source(source, registry)
            all_downloaded.extend(downloaded)
        except Exception as e:
            print(f"❌ Error processing {source.get('college')}: {e}")

    print(f"\n📦 Total new PDFs: {len(all_downloaded)}")

    if not all_downloaded:
        print("ℹ Nothing new to ingest.")
        return {"success": True, "ingested": 0}

    print("\n🔄 Running auto_ingest...")

    from ingestion.auto_ingest import extract_metadata, register_program

    success_count = 0

    for item in all_downloaded:
        pdf_path   = item["file_path"]
        college    = item["college"]
        country    = item["country"]
        state      = item["state"]
        source_url = item["source_url"]

        print(f"\n📄 {os.path.basename(pdf_path)}")

        try:
            use_filename_as_program = item.get("use_filename_as_program", False)

            if use_filename_as_program:
                raw_name = os.path.splitext(os.path.basename(pdf_path))[0]
                program  = unquote(raw_name.replace("_", " ").replace("-", " ").strip())
                degree_level = "UG"
            else:
                metadata = extract_metadata(pdf_path)
                program  = metadata.get("program", "").strip()
                if not program or program.lower() in ["na", "n/a", "not provided", ""]:
                    raw_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    program  = raw_name.replace("_", " ").replace("-", " ").strip()
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