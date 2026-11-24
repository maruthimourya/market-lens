"""
api.py — the FastAPI server, same pattern as convo-assistant.

For now it's a skeleton with one health-check endpoint.
We'll add the real endpoints as we build each piece.

Kotlin analogy: this is your Application class + Retrofit service definition.
  @GET("/health") fun health(): HealthResponse
"""

from fastapi import FastAPI

app = FastAPI(title="Market Lens", description="Stock-market teaching chatbot")


@app.get("/health")
def health() -> dict:
    """Smoke test: if this returns, the server is alive."""
    return {"status": "ok", "project": "market-lens"}
