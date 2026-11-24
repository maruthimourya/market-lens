"""
answer.py - generation: turn retrieved chunks into a CITED, HONEST explanation.

This closes the RAG loop:
  question -> [NewsIndex.search] -> [stuff chunks into a grounded prompt] -> [LLM]

THE BIG IDEA: the prompt IS the policy.
Every honesty rule we argued for becomes a line in the SYSTEM prompt:
  - ground ONLY in the provided articles (no outside memory)
  - explain PLAUSIBLE drivers, never assert certain causation
  - CITE the article titles
  - REFUSE (say so) when the articles don't explain the move
  - EDUCATIONAL only - never buy/sell/hold advice

THE PROMPT SANDWICH (same as convo-assistant):
  system  : the rules
  context : the retrieved chunks, pasted in
  question: the user's actual question
"""

from market_lens.chunking import NewsChunk
from market_lens.llm import make_llm
from market_lens.search import NewsIndex

SYSTEM = (
    "You are a stock-market EDUCATION assistant. You explain; you never advise.\n"
    "Answer the question using ONLY the news excerpts provided below.\n"
    "\n"
    "RULES:\n"
    "1. Explain PLAUSIBLE drivers of the move. Markets are noisy - never claim "
    "one thing definitely CAUSED another. Use hedged language: 'may have', "
    "'reportedly', 'around the same time', 'according to [source]'.\n"
    "2. Ground EVERY claim in the excerpts. Cite the article title in [brackets].\n"
    "3. If the excerpts do not explain the move, reply exactly: 'The sources I "
    "have don't clearly explain this move.' Never fill the gap from your own "
    "knowledge.\n"
    "4. NEVER recommend buying, selling, or holding. No price targets, no advice.\n"
    "Keep the answer to a short paragraph."
)


def explain(
    question: str, index: NewsIndex, k: int = 5
) -> tuple[str, list[tuple[NewsChunk, float]]]:
    """Retrieve the top k chunks, then have the LLM explain the move from them.

    Returns (explanation_text, hits) so the caller can show the answer AND the
    sources it was grounded in - the citation trail is the whole point.
    """
    hits = index.search(question, k=k)

    # Paste each retrieved chunk in, labelled with its article title.
    context = "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk, _ in hits)

    history = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"News excerpts:\n{context}\n\nQuestion: {question}",
        },
    ]
    reply = make_llm().reply(history)
    return reply, hits


# Demo: three questions that each test a different honesty rule.
if __name__ == "__main__":
    from market_lens.chunking import chunk_articles
    from market_lens.news import get_news

    articles = get_news("AAPL", from_date="2026-06-23", to_date="2026-06-25")
    index = NewsIndex(chunk_articles(articles))

    questions = [
        "Why did Apple stock fall on June 25, 2026?",  # 1. groundable -> explain + cite
        "Why did Tesla stock crash in 2024?",          # 2. not in sources -> refuse
        "Should I buy Apple stock right now?",         # 3. advice -> decline
    ]

    for q in questions:
        answer, _ = explain(q, index)
        print(f"Q: {q}")
        print(f"A: {answer}\n")
        print("-" * 70)
