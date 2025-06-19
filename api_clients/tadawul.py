import requests
from bs4 import BeautifulSoup
from api_clients.llm_api import chat_response
from utils.number_norm import normalize_numbers

def get_market_summary_report(limit=5):

    """
    Fetches the top `limit` gainers (English) and returns an empty losers list
    (the English page doesn't expose Top Losers in static HTML).
    """
    url = "https://www.argaam.com/en"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the Market Movers section
    header = soup.find("h2", string="Market Movers")
    if not header:
        raise RuntimeError("Could not locate the 'Market Movers' section on Argaam.")

    # Find the gainers table (the first table after header)
    table = header.find_next("table")
    if not table:
        raise RuntimeError("Could not find Market Movers table.")

    gainers = []
    rows = table.find_all("tr")[1: limit + 1]  # skip header row
    for row in rows:
        cols = [td.get_text(strip=True).replace("%", "") for td in row.find_all("td")]
        if len(cols) < 3:
            continue
        name = cols[0]
        try:
            pct = float(cols[2])
        except ValueError:
            continue
        gainers.append((name, pct))
    res = summarize_movers(gainers,"الأكثر ربحية")
    return res


def summarize_movers(movers: list[tuple[str, float]], title: str) -> str:
    """
    Given a list like [("سينومي ريتيل", 9.93), ...] and a title (e.g., "أعلى الرابحين"),
    prepares a prompt and calls chat_response() to produce a short Najdi-Arabic summary.
    """
    if not movers:
        return f"ما في {title.lower()} لليوم."

    # Build the prompt
    prompt = f"أنت مساعد سعودي يتكلم باللهجة النجدية.\nأعطيني نبذة عن {title} في السوق السعودي اليوم:\n\n"
    for idx, (name, pct) in enumerate(movers, start=1):
        prompt += f"{idx}. {name} (+{pct:.1f}  %)\n"
    prompt += f"\nسوِ لي هالملخص paragraph بسيط يناسب المستمعين السعوديين."
    prompt = normalize_numbers(prompt)

    print("📝 Prompt to GPT:", prompt)  # for debugging

    try:
        response = chat_response(prompt, [])
        return response

    except Exception as e:
        print(f"Summarization error: {e}")
        return f"عذرًا، ما قدرت ألخص لك {title.lower()}."


