from playwright.sync_api import sync_playwright

PAGES_TO_TEST = [
    ("SJSU CS Spring 2026", "https://www.sjsu.edu/cs/students/syllabi/spring-2026.php"),
]

WHITELIST = ["syllabus", "curriculum", "scheme", "course", "cs", "cse", "programming",
             "data", "algorithm", "operating", "network", "machine", "database", "software"]
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

            title = page.title()
            print(f"  Page title: {title}")

            all_links = page.eval_on_selector_all(
                "a[href]",
                "elements => elements.map(e => e.getAttribute('href'))"
            )
            print(f"  Total <a href> links found: {len(all_links)}")

            pdf_links = [l for l in all_links if l and ".pdf" in l.lower()]
            print(f"  PDF links found: {len(pdf_links)}")

            if pdf_links:
                print(f"\n  First 10 PDF links:")
                for l in pdf_links[:10]:
                    print(f"    {l}")
                print(f"\n  ✅ Playwright successfully scraped {len(pdf_links)} PDF links!")
            else:
                print(f"  ❌ No PDFs found")

            browser.close()

    except Exception as e:
        print(f"  ❌ Error: {e}")


for name, url in PAGES_TO_TEST:
    test_page(name, url)

print(f"\n{'='*60}")
print("Done.")