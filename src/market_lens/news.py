"""
news.py - fetch financial news for a ticker over a date range (station 3, part 1).

For a notable move on, say, 2026-06-25, we grab the news from around that date.
That news becomes the raw material the LLM is allowed to ground its explanation in.

Data source: EODHD news API. Needs a free EODHD_API_KEY in your .env.

WHY THE DISK CACHE:
  EODHD's free tier is rate-limited (a handful of calls per request, 20/day).
  So we save every response to data/cache/news/ and read from there on re-runs.
  Fetch once, re-read forever - we never pay for the same question twice.

  Kotlin view: this is a Room/OkHttp cache, hand-rolled.
    if (cache.exists(key)) cache.read(key) else api.fetch(key).also { cache.write(key, it) }
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://eodhd.com/api/news"
CACHE_DIR = Path("data/cache/news")


@dataclass
class NewsArticle:
    """One news article. (A Kotlin data class.)"""

    date: str  # ISO timestamp, e.g. "2026-06-25T13:30:00+00:00"
    title: str  # headline
    content: str  # full article body - what we'll chunk & ground on later
    link: str  # URL of the original article, for citing
    polarity: float  # sentiment score: -1 (very negative) .. +1 (very positive)


def _eodhd_symbol(ticker: str) -> str:
    """EODHD wants an exchange suffix: 'AAPL' -> 'AAPL.US'. Leave it if already set."""
    return ticker if "." in ticker else f"{ticker}.US"


def get_news(
    ticker: str, from_date: str, to_date: str, limit: int = 50
) -> list[NewsArticle]:
    """Fetch news for `ticker` between `from_date` and `to_date` (YYYY-MM-DD).

    Cached on disk, so the 2nd+ call for the same window is free and offline.
    """
    symbol = _eodhd_symbol(ticker)

    # --- disk cache: check for a saved answer before hitting the network ---
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{from_date}_{to_date}.json"

    if cache_file.exists():
        raw = json.loads(cache_file.read_text())  # free, offline re-read
    else:
        token = os.getenv("EODHD_API_KEY")
        if not token:
            raise RuntimeError("EODHD_API_KEY not set. Add it to your .env file.")

        resp = httpx.get(
            API_URL,
            params={
                "api_token": token,
                "s": symbol,
                "from": from_date,
                "to": to_date,
                "limit": limit,
                "fmt": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()  # turn an HTTP 4xx/5xx into a clear error
        raw = resp.json()
        cache_file.write_text(json.dumps(raw))  # save for next time

    # --- translate raw JSON -> our clean dataclass (same move as prices.py) ---
    articles: list[NewsArticle] = []
    for item in raw:
        sentiment = item.get("sentiment") or {}
        articles.append(
            NewsArticle(
                date=item.get("date", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                link=item.get("link", ""),
                polarity=float(sentiment.get("polarity", 0.0)),
            )
        )
    return articles


# Demo: fetch the news around AAPL's mystery move on 2026-06-25.
if __name__ == "__main__":
    articles = get_news("AAPL", from_date="2026-06-23", to_date="2026-06-25")
    print(f"Fetched {len(articles)} articles for AAPL around 2026-06-25\n")
    for a in articles[:5]:
        print(f"  [{a.date[:10]}] (sentiment {a.polarity:+.2f})  {a.title}")
