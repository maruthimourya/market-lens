"""
search.py - retrieval: given a question, find the chunks that best answer it.

Chunks are embedded IN MEMORY in a NewsIndex (built once, queried many times).

Two retrieval strategies:
  search(query, k)     -> plain top-k by relevance (cosine similarity)
  search_mmr(query, k) -> MMR rerank: relevant to the query AND diverse, so near
                          duplicates don't eat slots.

THE MATH: every chunk and the query become length-1 vectors. For length-1 vectors
dot product == cosine similarity. self.vectors @ query_vec scores the query against
ALL chunks at once.
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
    """An in-memory searchable index over one batch of chunks."""

    def __init__(self, chunks: list[NewsChunk]):
        self.chunks = chunks
        self.vectors = _model().encode(
            [c.text for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str, k: int = 5) -> list[tuple[NewsChunk, float]]:
        """Plain top-k: the k chunks most relevant to `query`, by cosine score."""
        query_vec = _model().encode(query, normalize_embeddings=True)
        scores = self.vectors @ query_vec
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def search_mmr(
        self, query: str, k: int = 5, lam: float = 0.6
    ) -> list[tuple[NewsChunk, float]]:
        """MMR rerank: relevant to the query, but NOT redundant with each other.

        Greedy. At each step pick the chunk that maximizes:
            lam * relevance(chunk, query) - (1 - lam) * max_similarity(chunk, picked)
        lam=1 -> pure relevance (same as search); lam=0 -> pure diversity.

        A near-duplicate has ~1.0 similarity to an already-picked chunk, so that
        penalty tanks its score and it never gets selected. Dedup for free.

        Kotlin feel: repeat(k) { pick the best-scoring remaining chunk }, where
        "best" blends a relevance score and a redundancy penalty.
        """
        query_vec = _model().encode(query, normalize_embeddings=True)
        relevance = self.vectors @ query_vec  # one relevance score per chunk

        candidates = list(range(len(self.chunks)))
        picked: list[int] = []
        while candidates and len(picked) < k:
            best_i, best_score = candidates[0], -1e9
            for i in candidates:
                # how redundant is chunk i with everything already picked?
                redundancy = 0.0
                if picked:
                    redundancy = max(
                        float(self.vectors[i] @ self.vectors[j]) for j in picked
                    )
                score = lam * float(relevance[i]) - (1 - lam) * redundancy
                if score > best_score:
                    best_score, best_i = score, i
            picked.append(best_i)
            candidates.remove(best_i)

        return [(self.chunks[i], float(relevance[i])) for i in picked]


# Demo: plain top-k vs MMR - watch the duplicate sources disappear.
if __name__ == "__main__":
    from market_lens.chunking import chunk_articles
    from market_lens.news import get_news

    articles = get_news("AAPL", from_date="2026-06-23", to_date="2026-06-26")
    index = NewsIndex(chunk_articles(articles))
    question = "Why did Apple stock fall on June 25, 2026?"

    def show(label, hits):
        unique = len({c.title for c, _ in hits})
        print(f"\n{label}   ({unique}/{len(hits)} UNIQUE articles)")
        for c, s in hits:
            print(f"  [{s:.3f}] {c.title[:60]}")

    print(f"Q: {question}")
    show("PLAIN top-k", index.search(question, k=5))
    show("MMR rerank (lam=0.6)", index.search_mmr(question, k=5, lam=0.6))
