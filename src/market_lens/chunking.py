"""
chunking.py - cut news articles into small, embeddable, CITEABLE pieces.

Adapts convo-assistant's chunker. The sliding-window core is copied verbatim;
what's new is handling NEWS instead of clean markdown:

  1. CLEAN the HTML cruft ('&amp;' -> '&', strip <p> tags) from BOTH the title
     and the body before embedding. Tags/entities are noise that blur an
     embedding (a meaning-vector), and the title is also what we cite.
  2. STAMP each chunk with its article's title + date + LINK, so the answer
     step can cite the exact source. No link = no citation = no trust.
  3. PREPEND the title so a mid-article slice still knows where it came from.

Kotlin view of a chunk:
  data class NewsChunk(val text: String, val title: String,
                       val date: String, val link: String, val index: Int)
"""

import html
import re
from dataclasses import dataclass

from market_lens.news import NewsArticle


@dataclass
class NewsChunk:
    text: str  # the piece we'll embed
    title: str  # article headline, cleaned (metadata)
    date: str  # article date, "2026-06-25" (metadata)
    link: str  # source URL - for citing
    index: int  # position among all chunks


def clean_html(raw: str) -> str:
    """Turn messy article HTML into plain text.

    - the regex deletes anything between < and > (tags like <p>, </a>)
    - html.unescape turns '&amp;' -> '&', '&#39;' -> "'"
    - the last regex squeezes runs of whitespace into single spaces
    Crude but dependency-free. A production app would reach for BeautifulSoup.
    """
    no_tags = re.sub(r"<[^>]+>", " ", raw)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Sliding window: cut a long string into overlapping pieces of <= max_chars.

    Copied straight from convo-assistant - a proven building block.
    Kotlin view: a window that steps forward by (max_chars - overlap).
    """
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap  # step back by `overlap` so windows share an edge
    return pieces


def chunk_articles(
    articles: list[NewsArticle], max_chars: int = 500, overlap: int = 50
) -> list[NewsChunk]:
    """Turn a list of news articles into a flat list of citeable chunks."""
    chunks: list[NewsChunk] = []
    index = 0
    for article in articles:
        title = clean_html(article.title)  # clean the title too, not just the body
        body = clean_html(article.content)
        if not body:
            continue  # skip empty articles
        # Prepend the (clean) title so each chunk carries its article's context.
        full_text = f"{title}. {body}"
        for piece in _split_long(full_text, max_chars, overlap):
            chunks.append(
                NewsChunk(
                    text=piece,
                    title=title,
                    date=article.date[:10],  # keep just YYYY-MM-DD
                    link=article.link,
                    index=index,
                )
            )
            index += 1
    return chunks


# Demo: fetch June 25 news, chunk it, and peek at the result.
if __name__ == "__main__":
    from market_lens.news import get_news

    articles = get_news("AAPL", from_date="2026-06-23", to_date="2026-06-25")
    chunks = chunk_articles(articles)
    print(f"{len(articles)} articles -> {len(chunks)} chunks\n")
    for c in chunks[:3]:
        print(f"[{c.date}]  {c.title[:55]}")
        print(f"   {c.text[:120]}...\n")
