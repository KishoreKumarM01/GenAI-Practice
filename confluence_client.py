import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

AUTH = HTTPBasicAuth(os.getenv("CONFLUENCE_EMAIL"), os.getenv("CONFLUENCE_API_TOKEN"))
BASE = os.getenv("CONFLUENCE_BASE")


def get_all_pages(space_key: str):
    pages, start = [], 0
    while True:
        res = requests.get(
            f"{BASE}/rest/api/content",
            params={
                "spaceKey": space_key,
                "expand": "body.export_view",
                "limit": 25,
                "start": start
            },
            auth=AUTH
        ).json()
        results = res.get("results", [])
        pages.extend(results)
        if len(results) < 25:
            break
        start += 25
    return pages


def get_page_by_title(title: str, space_key: str = None):
    params = {"title": title, "expand": "body.export_view"}
    if space_key:
        params["spaceKey"] = space_key
    res = requests.get(f"{BASE}/rest/api/content", params=params, auth=AUTH).json()
    results = res.get("results", [])
    return results[0] if results else None


def get_page_by_url(url: str):
    # Extract page ID from URL
    import re
    match = re.search(r'/pages/(\d+)', url)
    if not match:
        return None
    page_id = match.group(1)
    res = requests.get(
        f"{BASE}/rest/api/content/{page_id}",
        params={"expand": "body.export_view"},
        auth=AUTH
    )
    return res.json() if res.status_code == 200 else None


def get_page_title_from_url(url: str) -> str:
    page = get_page_by_url(url)
    return page.get("title", url) if page else url


def extract_text(page: dict) -> str:
    html = page.get("body", {}).get("export_view", {}).get("value", "")
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def get_page_metadata(page: dict) -> dict:
    return {
        "title": page.get("title", "Unknown"),
        "id": page.get("id", ""),
        "url": f"{BASE}/pages/{page.get('id', '')}"
    }
    
# quick test at bottom of file
if __name__ == "__main__":
    print(os.getenv("CONFLUENCE_EMAIL"))  # should print your email
    pages = get_all_pages(os.getenv("CONFLUENCE_SPACE_KEY"))
    print(f"Fetched {len(pages)} pages")
    print(extract_text(pages[0])[:500])