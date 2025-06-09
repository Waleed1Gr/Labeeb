# api_clients/news_api.py

from dotenv import load_dotenv
load_dotenv()

import requests
import feedparser
from api_clients.llm_api import chat_response

# ──────────────────────────────────────────────────────────────────────────────
# Okaz RSS feed (public, no 403 / valid XML):
#   https://www.okaz.com.sa/rssFeed/0
#
# We fetch it with a browser-like User-Agent, then parse with feedparser.
#───────────────────────────────────────────────────────────────────────────────

RSS_URL = "https://www.okaz.com.sa/rssFeed/0"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}

def fetch_okaz_headlines(n: int = 10) -> list[dict]:
    """
    Fetches the top n entries from Okaz’s RSS feed.
    Returns a list of dicts: [{"title": str, "url": str, "summary": str}, …]
    """
    try:
        # 1) Download RSS XML using a browser-like header
        resp = requests.get(RSS_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        xml_content = resp.content
    except Exception as e:
        print(f"News fetch error (network): {e}")
        return []

    try:
        # 2) Parse the raw XML with feedparser
        feed = feedparser.parse(xml_content)
    except Exception as e:
        print(f"News fetch error (feed parse failed): {e}")
        return []

    if feed.bozo:
        # bozo == True indicates a parse error inside feedparser
        print(f"News fetch error: RSS malformed or feedparser error ({feed.bozo_exception})")
        return []

    # 3) Take the first n entries
    entries = feed.entries[:n]
    results = []
    for entry in entries:
        title = entry.get("title", "").strip()
        url   = entry.get("link", "").strip()
        summary_html = entry.get("summary", "").strip()
        # Optionally strip HTML tags if needed:
        # from bs4 import BeautifulSoup
        # summary = BeautifulSoup(summary_html, "html.parser").get_text().strip()
        summary = summary_html

        results.append({"title": title, "url": url, "summary": summary})
    return results


def summarize_headlines(headlines: list[dict]) -> str:
    """
    Given a list of {"title": ..., "url": ..., "summary": ...},
    send them to GPT via chat_response() to produce a Najdi-Arabic summary.
    """
    if not headlines:
        return "عذرًا، ما لقيت أي أخبار جديدة الآن."

    prompt_text = "أنت مساعد إخباري سعودي. هذه أحدث أخبار عكاظ:\n\n"
    for idx, item in enumerate(headlines, start=1):
        t = item["title"]
        s = item["summary"] or "[لا يوجد ملخص متاح]"
        prompt_text += f"{idx}. العنوان: {t}\n   موجز الخبر: {s}\n\n"

    prompt_text += "أولخص لي هذه الأخبار باللهجة النجدية في فقرة قصيرة تناسب المستمعين السعوديين."

    try:
        response = chat_response(prompt_text, [])
        return response.strip()
    except Exception as e:
        print(f"News summarization error: {e}")
        return "عذرًا، حصل خطأ أثناء تلخيص الأخبار."
