"""
evaluate.py - measure FAITHFULNESS: does every claim in an answer trace to a source?

OFFLINE evaluation - a measuring stick for development, not live code.

TWO-STEP LLM-as-judge:
  1. extract_claims(answer)          -> standalone factual claims   [1 judge call]
  2. verify_claims(claims, sources)  -> supported? yes/no for each   [1 judge call]
  faithfulness = supported / total

THE JUDGE IS A DIFFERENT MODEL THAN THE GENERATOR (don't let a model grade its
own homework). Generator = llama-3.3-70b; judge = gpt-oss-20b (different lineage).

NOTE: gpt-oss is a REASONING model - it spends tokens "thinking" before it writes
the answer. A small token budget gets eaten by the thinking and leaves the content
EMPTY, so the judge needs a big max_tokens. Override the model with JUDGE_MODEL.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")


def _ask_judge(system: str, user: str) -> str:
    """Call the judge directly with a BIG token budget - reasoning models think
    before answering, and a small budget leaves the content empty."""
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=2048,  # room to think AND answer
    )
    return resp.choices[0].message.content or ""


# ---------- step 1: pull the answer apart into individual claims ----------
EXTRACT_SYSTEM = (
    "Break the text into its individual factual claims. Each claim must be a "
    "single, standalone statement. Drop hedging words like 'possibly' or "
    "'may have'. Return ONLY a JSON array of strings - nothing else."
)


def extract_claims(answer: str) -> list[str]:
    return _parse_json(_ask_judge(EXTRACT_SYSTEM, answer))


# ---------- step 2: fact-check every claim against the sources, in one call ----------
VERIFY_SYSTEM = (
    "You are a strict fact-checker. For EACH numbered claim, decide if it is "
    "SUPPORTED using ONLY the sources provided - never outside knowledge. "
    "Return ONLY a JSON array, one object per claim IN ORDER:\n"
    '{"claim": "<the claim>", "supported": true or false, "why": "<short reason>"}'
)


def verify_claims(claims: list[str], sources: str) -> list[dict]:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    user = f"Sources:\n{sources}\n\nClaims:\n{numbered}"
    return _parse_json(_ask_judge(VERIFY_SYSTEM, user))


# A refusal makes no claims, so faithfulness does not apply - never score it.
REFUSAL_MARKERS = ("don't clearly explain", "do not clearly explain",
                   "don't have a lesson", "i don't have", "i do not have")


def _is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def faithfulness(answer: str, sources: str) -> dict:
    """Return a faithfulness report: score + per-claim verdicts (never crashes)."""
    if _is_refusal(answer):
        return {"score": None, "supported": 0, "total": 0, "verdicts": [],
                "note": "answer was a refusal - no claims to fact-check"}
    claims = extract_claims(answer)
    if not claims:
        return {"score": None, "supported": 0, "total": 0, "verdicts": [],
                "note": "judge extracted no claims"}
    verdicts = verify_claims(claims, sources)
    if not verdicts:
        return {"score": None, "supported": 0, "total": 0, "verdicts": [],
                "note": "judge returned no verdicts"}
    supported = sum(1 for v in verdicts if v.get("supported"))
    return {
        "score": supported / len(verdicts),
        "supported": supported,
        "total": len(verdicts),
        "verdicts": verdicts,
        "note": "",
    }


def _parse_json(text: str):
    """LLMs wrap JSON in ```fences``` or add chatter. Grab the outermost [...]."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


# Demo: score a REAL answer from the move pipeline, claim by claim.
if __name__ == "__main__":
    from market_lens.chunking import chunk_articles
    from market_lens.news import get_news
    from market_lens.search import NewsIndex

    articles = get_news("AAPL", from_date="2026-06-23", to_date="2026-06-26")
    index = NewsIndex(chunk_articles(articles))
    hits = index.search("Why did Apple stock fall on June 25, 2026?", k=5)
    sources = "\n\n".join(c.text for c, _ in hits)

    # A FIXED sample answer (a real one the generator produced earlier) so the
    # demo reliably shows the judge at work, embellishments included. In real use
    # you'd pass a live answer from the pipeline here instead.
    answer = (
        "Apple shares fell on June 25, 2026 as the market reacted to news that "
        "Apple was raising prices on Macs, iPads and accessories by 15% to 25% "
        "amid rising component costs. The price hikes were described as "
        "unprecedented over the past 15 years by Morgan Stanley's Erik Woodring, "
        "and the stock dropped about 6% on the day."
    )

    print(f"JUDGE MODEL: {JUDGE_MODEL}  (generator: llama-3.3-70b-versatile)\n")
    print("ANSWER UNDER TEST:")
    print(answer, "\n")

    report = faithfulness(answer, sources)
    if report["score"] is None:
        print(f"COULD NOT SCORE: {report['note']}")
    else:
        print(
            f"FAITHFULNESS: {report['supported']}/{report['total']} claims supported "
            f"(score {report['score']:.2f})\n"
        )
        for v in report["verdicts"]:
            mark = "OK" if v.get("supported") else "!!"
            print(f"  [{mark}] {v.get('claim', '')[:72]}")
            if not v.get("supported"):
                print(f"        reason: {v.get('why', '')[:80]}")
