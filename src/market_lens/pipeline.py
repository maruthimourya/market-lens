"""
pipeline.py - the orchestrator: given a ticker, explain its notable moves.

Ties every station together into one call:
  ticker -> get_prices -> notable_moves
         -> for each move: news(D-2 .. D+1) -> chunk -> NewsIndex -> explain

This is the whole Phase 1 "explain this move" loop, generalized from one
hardcoded stock+date to any ticker.
"""

from datetime import date, timedelta

from market_lens.answer import explain
from market_lens.chunking import chunk_articles
from market_lens.news import get_news
from market_lens.prices import get_prices, notable_moves
from market_lens.search import NewsIndex


def _news_window(move_date: str, days_before: int = 2, days_after: int = 1) -> tuple[str, str]:
    """News window around a move: D-days_before .. D+days_after.

    We include D+1 because the "here's why it fell" explainer articles often
    get published the morning AFTER the move. Cost: a bit more noise + quota.
    """
    d = date.fromisoformat(move_date)
    start = d - timedelta(days=days_before)
    end = d + timedelta(days=days_after)
    return start.isoformat(), end.isoformat()


def explain_moves(ticker: str, period: str = "6mo", threshold_pct: float = 5.0) -> list[dict]:
    """For each notable move of `ticker`, fetch surrounding news and explain it."""
    prices = get_prices(ticker, period=period)
    moves = notable_moves(prices, threshold_pct=threshold_pct)

    results: list[dict] = []
    for move in moves:
        from_date, to_date = _news_window(move.date)
        articles = get_news(ticker, from_date=from_date, to_date=to_date)
        if not articles:
            continue  # no news that window -> skip rather than invent

        index = NewsIndex(chunk_articles(articles))
        direction = "rise" if move.pct_change > 0 else "fall"
        question = f"Why did {ticker} stock {direction} on {move.date}?"
        answer, hits = explain(question, index)

        results.append(
            {"move": move, "question": question, "answer": answer, "hits": hits}
        )
    return results


if __name__ == "__main__":
    ticker = "AAPL"
    print(f"Explaining notable moves for {ticker}...\n")
    for r in explain_moves(ticker):
        m = r["move"]
        print("=" * 70)
        print(f"{m.date}:  {m.pct_change:+.1f}%   (close ${m.close})")
        print("=" * 70)
        print(r["answer"])
        print()
