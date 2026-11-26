"""
concepts.py - load the investing-concepts glossary and make it searchable.

The teaching chat answers "what is a P/E ratio?" by GROUNDING in this corpus -
the same discipline as move questions, just a different shelf (Investor.gov
instead of news).

The corpus is one markdown file, one concept per "## heading". Each concept is
short, so one concept = one chunk (no sliding-window splitting needed here).

We REUSE NewsIndex from search.py. It only reads `.text` off each chunk, so any
object with a .text attribute works - including ConceptChunk. That's "duck
typing": Python checks "does it have .text?" at runtime, where Kotlin would make
you declare a shared interface up front.
"""

from dataclasses import dataclass
from pathlib import Path

CONCEPTS_PATH = Path("data/concepts/investor_gov_glossary.md")
SOURCE_NAME = "Investor.gov (SEC)"
SOURCE_URL = "https://www.investor.gov/introduction-investing/investing-basics/glossary"


@dataclass
class ConceptChunk:
    text: str  # "Term. definition" - what gets embedded
    term: str  # the concept name (metadata)
    source: str  # human-readable source name (for citing)
    link: str  # source URL (for citing)


def load_concepts(path: Path = CONCEPTS_PATH) -> list[ConceptChunk]:
    """Parse the glossary markdown into one ConceptChunk per '## heading'."""
    md = path.read_text()
    chunks: list[ConceptChunk] = []
    for block in md.split("\n## "):
        head, _, body = block.partition("\n")
        term = head.strip()
        body = body.strip()
        # skip the file's title/source block (starts with '#') and empties
        if term.startswith("#") or not body:
            continue
        chunks.append(
            ConceptChunk(
                text=f"{term}. {body}",
                term=term,
                source=SOURCE_NAME,
                link=SOURCE_URL,
            )
        )
    return chunks


# Demo: load the concepts and retrieve the best match for a question.
if __name__ == "__main__":
    from market_lens.search import NewsIndex

    chunks = load_concepts()
    print(f"Loaded {len(chunks)} concepts\n")

    index = NewsIndex(chunks)  # reused as-is - ConceptChunk has .text
    for chunk, score in index.search("what is a p/e ratio?", k=3):
        print(f"[{score:.3f}] {chunk.term}")
