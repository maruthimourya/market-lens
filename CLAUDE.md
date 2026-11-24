# Market Lens

A stock-market **teaching** chatbot that explains price moves using grounded sources.
Educational tool, **NOT financial advice**: it surfaces what happened around notable
price moves (earnings, news, SEC filings) and explains *plausible* drivers — it never
recommends buys/sells, and it cites every claim.

## Who I'm working with

- Principal mobile developer (Kotlin/Android), moving toward an AI/ML engineer role.
- New to Python and ML. Treats `convo-assistant` (a sibling repo) as a learning reference.
- Learning by building this project step by step.

## How to teach me (follow in every reply)

- Explain new concepts at a **beginner level** — the idea, not just the command to run.
- Put a **short parenthetical definition** next to jargon, e.g.
  "embedding (a list of numbers that represents meaning)".
- **Compare everything to Kotlin/Android** — it's how I map new ideas.
- Keep sentences simple. If I say I'm lost, stop and re-explain with a smaller example.
- Some **humor** is welcome.
- **Surprise pop quizzes** at checkpoints — application and trade-off questions, not
  multiple-choice recognition. One at a time; wait for my answer, evaluate it, move on.
- When changing my Mac / project setup, **explain what each command does and why**.
- **Wait for my go-ahead before installing packages or editing `pyproject.toml`.**

## Project conventions

- **One file, one job** — flat `src/market_lens/`, no premature subpackages. Simpler is better.
- Reuse `convo-assistant`'s proven building blocks by **copying**, not cross-importing.
- **Honesty discipline is the whole point:** ground answers only in retrieved sources,
  explain *plausible* (not certain) drivers, cite sources, refuse when the sources don't
  explain a move, and never give buy/sell advice. These rules live in the SYSTEM prompt
  in `answer.py`.

## Architecture (Phase 1 — the "explain this move" RAG loop)

Pipeline, data flowing left to right:

- `prices.py`   — daily prices (yfinance) + flag notable moves (>= 5% swing)
- `news.py`     — news around a move (EODHD), cached to disk
- `chunking.py` — clean HTML + cut articles into citeable, embeddable chunks
- `search.py`   — embed chunks, retrieve the most relevant (NewsIndex)
- `answer.py`   — ground + generate a cited, hedged explanation (or refuse)
- `pipeline.py` — orchestrator: ticker -> moves -> explain each (news window D-2..D+1)
- `llm.py`      — Groq (cloud) / Ollama (local) providers
- `api.py`      — FastAPI server (skeleton)

Run any module's demo: `uv run python -m market_lens.<module>`

## Roadmap

- **Phase 1 (now):** ~10 big companies, the explain-this-move loop + a teaching chat.
- **Phase 2:** scale toward top 100, add a reranker (dedup/MMR), a real vector DB,
  RAGAS-style evaluation (faithfulness), boilerplate stripping.
- **Phase 3:** Kotlin/Android mobile app.

## Secrets

`GROQ_API_KEY` and `EODHD_API_KEY` live in `.env` (git-ignored). Never commit them.
