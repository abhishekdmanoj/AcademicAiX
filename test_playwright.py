from playwright.sync_api import sync_playwright

PAGES_TO_TEST = [
    ("NIT Trichy", "https://www.nitt.edu/home/academics/departments/cse/curriculum_and_syllabus/"),
    ("IIT Bombay", "https://www.cse.iitb.ac.in/academics/ugcurriculum.php"),
    ("IIT Kharagpur", "https://erp.iitkgp.ac.in/ERPWebServices/curricula/CurriculaSubjectsList.jsp?stuType=UG"),
]

WHITELIST = ["syllabus", "curriculum", "scheme", "course"]
BLACKLIST = ["fee", "timetable", "admission", "hostel", "tender", "notice"]


def test_page(name, url):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            print("  Navigating...")
            page.goto(url, timeout=30000)
            page.wait_for_load_state("networkidle", timeout=15000)

            # Page title — confirms we actually landed somewhere
            title = page.title()
            print(f"  Page title: {title}")

            # Total link count
            all_links = page.eval_on_selector_all(
                "a[href]",
                "elements => elements.map(e => e.getAttribute('href'))"
            )
            print(f"  Total <a href> links found: {len(all_links)}")

            if len(all_links) == 0:
                print("  ⚠️  Zero links — page may be fully JS-rendered with no anchor tags")
                # Dump raw HTML snippet to help diagnose
                html = page.content()
                print(f"  HTML length: {len(html)} chars")
                print(f"  HTML preview (first 500 chars):")
                print(f"    {html[:500]}")
            else:
                # Show first 10 hrefs so we can see what's there
                print(f"  Sample hrefs (first 10):")
                for l in all_links[:10]:
                    print(f"    {l}")

                # PDF links
                pdf_links = [l for l in all_links if l and ".pdf" in l.lower()]
                print(f"\n  PDF links found: {len(pdf_links)}")
                for l in pdf_links[:10]:
                    print(f"    {l}")

                # Filtered
                filtered = [
                    l for l in pdf_links
                    if any(w in l.lower() for w in WHITELIST)
                    and not any(w in l.lower() for w in BLACKLIST)
                ]
                print(f"  After syllabus filter: {len(filtered)}")
                for l in filtered[:5]:
                    print(f"    ✅ {l}")

                if filtered:
                    print(f"\n  ✅ PLAYWRIGHT WORKS — syllabus PDFs found")
                elif pdf_links:
                    print(f"\n  ⚠️  PDFs found but none match syllabus filter")
                else:
                    print(f"\n  ❌ No PDFs found — check if content loads differently")

            browser.close()

    except Exception as e:
        print(f"  ❌ Error: {e}")


for name, url in PAGES_TO_TEST:
    test_page(name, url)

print(f"\n{'='*60}")
print("Done.")