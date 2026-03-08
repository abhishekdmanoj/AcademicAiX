import requests
from bs4 import BeautifulSoup

PAGES_TO_TEST = [
    "https://www.nitt.edu/home/academics/departments/cse/curriculum_and_syllabus/",
    "https://www.iitb.ac.in/academics/ugcurriculum.php",
    "https://erp.iitkgp.ac.in/ERPWebServices/curricula/CurriculaSubjectsList.jsp?stuType=UG",
]

WHITELIST = ["syllabus", "curriculum", "scheme", "course"]
BLACKLIST = ["fee", "timetable", "admission", "hostel", "tender", "notice"]

def test_page(url):
    print(f"\nTesting: {url}")
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        all_links = [a["href"] for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]
        print(f"  Total PDF links found: {len(all_links)}")

        filtered = []
        for link in all_links:
            lower = link.lower()
            if any(w in lower for w in WHITELIST) and not any(w in lower for w in BLACKLIST):
                filtered.append(link)

        print(f"  After whitelist/blacklist filter: {len(filtered)}")
        for l in filtered[:5]:
            print(f"    {l}")

        if not filtered:
            print("  ❌ No matching PDFs — page scraper won't work here")
        else:
            print("  ✅ Page scraper will work here")

    except Exception as e:
        print(f"  ❌ Error: {e}")

for page in PAGES_TO_TEST:
    test_page(page)
