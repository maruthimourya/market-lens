"""
chunking.py - cut news articles into small, embeddable, CITEABLE pieces.

Adapts convo-assistant's chunker. The sliding-window core is copied verbatim;
what's new is handling NEWS instead of clean markdown:

  1. CLEAN the HTML cruft ('&amp;' -> '&', strip <p> tags) from title + body.
  2. STRIP promotional/disclaimer BOILERPLATE (publisher junk that isn't news).
  3. STAMP each chunk with its article's title + date + LINK (for citing).
  4. PREPEND the title to EVERY chunk's text (not just the first), so every
     slice the embedder sees knows which article it came from.

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


# Promo / disclaimer phrases that mark a SENTENCE as boilerplate, not news.
# Matched case-insensitively. This is a LIVING list: add a publisher's markers
# the first time its junk shows up (the price of the simple rule-based approach).
BOILERPLATE_MARKERS = (
    "the motley fool",
    "stock advisor",
    "double down",
    "disclosure policy",
    "10 best stocks",
    "best stocks to buy",
    "see the 3 stocks",
    "join stock",
)


def clean_html(raw: str) -> str:
    """Turn messy article HTML into plain text.

    - the regex deletes anything between < and > (tags like <p>, </a>)
    - html.unescape turns '&amp;' -> '&', '&#39;' -> "'"
    - the last regex squeezes runs of whitespace into single spaces
    """
    no_tags = re.sub(r"<[^>]+>", " ", raw)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def strip_boilerplate(text: str) -> str:
    """Remove promo/disclaimer sentences; keep the real article.

    SURGICAL (option b): split into sentences, drop only the sentences that
    carry a boilerplate marker, rejoin the rest. Content BEFORE and AFTER the
    junk survives - which matters because publishers often put promo ABOVE the
    article, so cutting 'from the marker to the end' would delete real news.

    Kotlin view:
      sentences.filter { s -> markers.none { s.contains(it, ignoreCase = true) } }
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [
        s
        for s in sentences
        if not any(marker in s.lower() for marker in BOILERPLATE_MARKERS)
    ]
    return " ".join(kept).strip()


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
        title = clean_html(article.title)
        body = clean_html(article.content)
        body = strip_boilerplate(body)  # drop publisher promo/disclaimer junk
        if not body:
            continue  # nothing left but boilerplate -> skip

        # Put the title on EVERY chunk, not just the first. Reserve room for the
        # title (its ACTUAL length), split the BODY into what's left, then stamp
        # the title onto each slice. So each chunk = title + body-slice, and it
        # still fits in max_chars - keeping every embedding small and sharp.
        # Kotlin: body.chunked(budget).map { "$title. $it" }
        prefix = f"{title}. "
        budget = max(max_chars - len(prefix), 100)  # room left for the body
        for piece in _split_long(body, budget, overlap):
            chunks.append(
                NewsChunk(
                    text=prefix + piece,
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
