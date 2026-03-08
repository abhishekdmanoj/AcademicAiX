import requests
from bs4 import BeautifulSoup

url = "https://www.fau.edu/engineering/eecs/undergraduate/syllabi-ce-cs/"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

r = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(r.text, "html.parser")

all_links = [a["href"] for a in soup.find_all("a", href=True)]
pdf_links = [l for l in all_links if ".pdf" in l.lower()]

print(f"Total links on page: {len(all_links)}")
print(f"PDF links found: {len(pdf_links)}")
for l in pdf_links:
    print(f"  {l}")
