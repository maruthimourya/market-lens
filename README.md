# Market Lens

A stock-market teaching chatbot that explains price moves using grounded sources.

**Educational tool, NOT financial advice.** It surfaces what happened around
notable price moves (earnings, news, SEC filings) and explains plausible
drivers — it never recommends buys or sells.

## Setup

```bash
uv sync                    # install dependencies
cp .env.example .env       # then fill in your GROQ_API_KEY
```

## Run

```bash
uv run uvicorn market_lens.api:app --reload
```
