r"""
chat.py - the teaching chat: route a question to the right grounded source.

ROUTING is intent classification - the same job you did in convo-assistant with
TF-IDF / DistilBERT, here done with a one-line LLM call:

  question --> [route] --> CONCEPT -> concepts glossary (Investor.gov)
                       \-> MOVE    -> news pipeline

Both paths follow the SAME discipline: ground only in the retrieved source, cite
it, refuse if it's not covered, and never give buy/sell advice.
"""

from functools import lru_cache

from market_lens.answer import explain
from market_lens.chunking import chunk_articles
from market_lens.concepts import load_concepts
from market_lens.llm import make_llm
from market_lens.news import get_news
from market_lens.search import NewsIndex

# ---------- the router (intent classifier) ----------
ROUTER_SYSTEM = (
    "Classify the question into ONE word.\n"
    "MOVE = asks why a specific stock's price moved (up or down) on a date or recently.\n"
    "CONCEPT = asks about a general investing idea, term, or definition.\n"
    "Reply with ONLY the single word MOVE or CONCEPT."
)


def route(question: str) -> str:
    reply = (
        make_llm()
        .reply(
            [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": question},
            ]
        )
        .strip()
        .upper()
    )
    return "MOVE" if "MOVE" in reply else "CONCEPT"


# ---------- concept path: ground in the Investor.gov glossary ----------
CONCEPT_SYSTEM = (
    "You are a stock-market EDUCATION assistant for a beginner. Explain the "
    "concept simply, using ONLY the glossary entries provided.\n"
    "Rules:\n"
    "1. Ground every claim in the entries. Cite the source as [Investor.gov].\n"
    "2. If the entries do not cover the question, say exactly: 'I don't have a "
    "lesson on that yet.'\n"
    "3. Never give buy/sell/hold advice.\n"
    "Keep it to a short, friendly paragraph."
)


@lru_cache(maxsize=1)
def _concept_index() -> NewsIndex:
    """Build the concepts index once, then reuse it."""
    return NewsIndex(load_concepts())


def answer_concept(question: str, k: int = 3):
    hits = _concept_index().search(question, k=k)
    context = "\n\n".join(f"[{c.term}] {c.text}" for c, _ in hits)
    reply = make_llm().reply(
        [
            {"role": "system", "content": CONCEPT_SYSTEM},
            {
                "role": "user",
                "content": f"Glossary entries:\n{context}\n\nQuestion: {question}",
            },
        ]
    )
    return reply, hits


# ---------- move path: reuse the grounded news loop ----------
KNOWN_TICKERS = {
    "AAPL": ("AAPL", "APPLE"),
    "MSFT": ("MSFT", "MICROSOFT"),
    "NVDA": ("NVDA", "NVIDIA"),
}


def find_ticker(question: str) -> str:
    up = question.upper()
    for ticker, names in KNOWN_TICKERS.items():
        if any(name in up for name in names):
            return ticker
    return "AAPL"  # sensible default for the demo


def answer_move(question: str, window: tuple[str, str] = ("2026-06-23", "2026-06-26")):
    # NOTE: the news window is fixed to our cached demo data for now. Pulling the
    # relevant DATE out of the question (NL date parsing) is the next step.
    ticker = find_ticker(question)
    articles = get_news(ticker, from_date=window[0], to_date=window[1])
    index = NewsIndex(chunk_articles(articles))
    return explain(question, index)


# ---------- the chat entry point ----------
def teach(question: str):
    """Route the question, answer from the right grounded source.

    Returns (kind, answer_text, sources) where sources is a list of (label, url).
    """
    kind = route(question)
    if kind == "CONCEPT":
        reply, hits = answer_concept(question)
        get_label = lambda c: (c.source, c.link)
    else:
        reply, hits = answer_move(question)
        get_label = lambda c: (c.title, c.link)

    seen: set = set()
    sources: list = []
    for chunk, _ in hits:
        label, url = get_label(chunk)
        if (label, url) not in seen:          # skip duplicate sources
            seen.add((label, url))
            sources.append((label, url))
    return kind, reply, sources


if __name__ == "__main__":
    questions = [
        "What is a P/E ratio?",  # CONCEPT -> Investor.gov
        "Why did Apple stock fall on June 25, 2026?",  # MOVE -> news
    ]
    for q in questions:
        kind, answer, sources = teach(q)
        tag = "CONCEPT -> Investor.gov" if kind == "CONCEPT" else "MOVE -> news"
        print("=" * 70)
        print(f"Q  [{tag}]")
        print(f"   {q}")
        print("=" * 70)
        print(answer)
        print("\nSources:")
        for label, url in sources:
            print(f"  - {label}\n    {url}")
        print()
