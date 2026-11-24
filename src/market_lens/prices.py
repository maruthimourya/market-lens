"""
prices.py - fetch a stock's daily prices, then flag the "notable" moves.

This is stations 1 + 2 of the pipeline:
    ticker --> [download price history] --> [spot the big moves]

NO LLM here. This file is pure data: numbers in, numbers out. That makes it
easy to test and easy to trust - the foundation everything else sits on.

TWO NEW TOOLS:
  yfinance : downloads daily prices from Yahoo Finance. Like a Retrofit client
             someone already wrote for you.
  DataFrame: the table yfinance hands back (from the pandas library). We touch
             it for 3 lines, grab what we need, and escape into plain Python.

Kotlin view of what we produce:
  data class PriceDay(val date: String, val close: Double, val pctChange: Double)
  fun getPrices(ticker: String): List<PriceDay>
  fun notableMoves(prices: List<PriceDay>): List<PriceDay>
"""

from dataclasses import dataclass

import yfinance as yf


@dataclass
class PriceDay:
    """One trading day for one stock. (A Kotlin data class, basically.)"""

    date: str  # "2026-01-15"
    close: float  # closing price in dollars, e.g. 189.42
    pct_change: float  # % change vs the PREVIOUS day's close, e.g. +6.5


def get_prices(ticker: str, period: str = "6mo") -> list[PriceDay]:
    """Download `period` of daily prices for `ticker`, as a list of PriceDays.

    period examples: "1mo", "6mo", "1y", "5y".
    """
    # --- the only 3 lines that touch pandas ---
    data = yf.Ticker(ticker).history(period=period)  # a DataFrame (table)
    dates = [d.strftime("%Y-%m-%d") for d in data.index]  # row labels are dates
    closes = data["Close"].tolist()  # the "Close" column -> plain Python list
    # ------------------------------------------

    # Now walk the days in order, computing each day's % change from the one before.
    # (The first day has nothing before it, so its change is 0.)
    days: list[PriceDay] = []
    prev_close: float | None = None
    for date, close in zip(dates, closes):
        if prev_close is None:
            pct = 0.0
        else:
            pct = (close - prev_close) / prev_close * 100
        days.append(PriceDay(date=date, close=round(close, 2), pct_change=round(pct, 2)))
        prev_close = close

    return days


def notable_moves(prices: list[PriceDay], threshold_pct: float = 5.0) -> list[PriceDay]:
    """Keep only the days that moved at least `threshold_pct` in EITHER direction.

    abs() so a -6% crash counts as notable just like a +6% jump.

    Kotlin view:
      prices.filter { abs(it.pctChange) >= thresholdPct }
    """
    return [day for day in prices if abs(day.pct_change) >= threshold_pct]


# A tiny demo so you can run this file on its own and SEE it work.
# Runs only when you execute the file directly, not when another file imports it.
# (Like `fun main()` guarded so it doesn't fire when the class is used as a library.)
if __name__ == "__main__":
    prices = get_prices("AAPL", period="6mo")
    print(f"Fetched {len(prices)} trading days for AAPL\n")

    moves = notable_moves(prices, threshold_pct=5.0)
    print(f"Found {len(moves)} notable moves (>= 5%):")
    for m in moves:
        print(f"  {m.date}:  {m.pct_change:+.1f}%   (close ${m.close})")
