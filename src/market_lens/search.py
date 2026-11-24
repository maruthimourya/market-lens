"""
search.py - retrieval: given a question, find the chunks that best answer it.

convo-assistant's semantic search, with ONE change: convo-assistant precomputed
its chunk vectors to disk because its documents were fixed. Our news is fetched
per move, so we embed the chunks IN MEMORY and keep them in a NewsIndex object -
build the index once, then ask it many questions.

THE MATH (same as convo-assistant):
  Every chunk and the query become length-1 vectors (embeddings). For length-1
  vectors, dot product == cosine similarity (how aligned two meanings are).
  self.vectors @ query_vec dot-products the query against ALL chunks at once,
  giving one similarity score per chunk. Sort, take the top k.

Kotlin view:
  class NewsIndex(chunks) {
    val vectors = chunks.map { embed(it.text) }          // state
    fun search(q: String, k: Int) =                       // behavior
      chunks.indices.sortedByDescending { cosine(vectors[it], embed(q)) }.take(k)
  }
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from market_lens.chunking import NewsChunk

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model once, then reuse it (loading is slow)."""
    return SentenceTransformer(MODEL_NAME)


class NewsIndex:
    """An in-memory searchable index over one batch of news chunks.

    Build it once (embeds every chunk), then call .search() as often as you like.
    """

    def __init__(self, chunks: list[NewsChunk]):
        self.chunks = chunks
        # Embed every chunk's text into a length-1 vector. Done ONCE, up front.
        self.vectors = _model().encode(
            [c.text for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str, k: int = 5) -> list[tuple[NewsChunk, float]]:
        """Return the k chunks most relevant to `query`, each with its score."""
        query_vec = _model().encode(query, normalize_embeddings=True)

        # One dot product against the whole stack -> one score per chunk.
        scores = self.vectors @ query_vec

        # Pure-Python sort of the chunk indices by score, best first, take k.
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


# Demo: the first time the pipeline RETRIEVES an answer to a real question.
if __name__ == "__main__":
    from market_lens.chunking import chunk_articles
    from market_lens.news import get_news

    articles = get_news("AAPL", from_date="2026-06-23", to_date="2026-06-25")
    chunks = chunk_articles(articles)
    index = NewsIndex(chunks)  # embeds all 317 chunks

    question = "Why did Apple stock fall on June 25?"
    print(f"Q: {question}\n")
    for chunk, score in index.search(question, k=5):
        print(f"[{score:.3f}] ({chunk.date}) {chunk.title[:50]}")
        print(f"   {chunk.text[:140]}...\n")
