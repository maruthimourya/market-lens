"""
llm.py — the "brain" layer: talk to an LLM via Groq (cloud) or Ollama (local).

Reused from convo-assistant with zero changes to the interface.
Any module that needs an LLM calls make_llm() and gets back an object
whose .reply(history) method returns a string.

Kotlin analogy:
  interface LLM { fun reply(history: List<Message>): String }
  fun makeLLM(): LLM = when (BuildConfig.LLM_PROVIDER) { "groq" -> GroqLLM() ... }
"""

import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLM(Protocol):
    """The interface: anything with a reply() method counts as an LLM."""

    def reply(self, history: list[dict]) -> str: ...


class GroqLLM:
    """Cloud brain: a model hosted on Groq's servers. Needs GROQ_API_KEY."""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.model = model

    def reply(self, history: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=history, max_tokens=800
        )
        return resp.choices[0].message.content or ""


class OllamaLLM:
    """Local brain: a model running on your Mac through Ollama. Free, offline."""

    def __init__(self, model: str = "qwen3:8b"):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model = model

    def reply(self, history: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=history, max_tokens=800
        )
        return resp.choices[0].message.content or ""


def make_llm() -> LLM:
    """Factory: picks the brain from the LLM_PROVIDER env var (default: groq)."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        return GroqLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use 'groq' or 'ollama')")
