"""
ISDO - Knowledge Base Indexer
-----------------------------
1. Reads all .md files from data/kb/
2. Splits each article into chunks at '## ' (level-2) headings
3. Stores the chunks in a ChromaDB collection called 'isdo_kb'
4. Runs sample queries and prints the best-matching article + confidence

Dependency: chromadb only (uses Chroma's built-in default embedding model,
all-MiniLM-L6-v2 via ONNX; it downloads ~80 MB on the very first run).

Run from the project root:
    python build_kb_index.py
"""

import re
from pathlib import Path

import chromadb

def _find_project_root() -> Path:
    """Walk up from this script's folder until a folder containing data/kb is found."""
    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents]:
        if (folder / "data" / "kb").is_dir():
            return folder
    raise FileNotFoundError(f"Could not find a 'data/kb' folder above {here}")


BASE_DIR = _find_project_root()
KB_DIR = BASE_DIR / "data" / "kb"
DB_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "VPN says authentication failed after I changed my password",
    "Outlook on my phone is not syncing emails",
    "SAP login error DBCON_FAIL for the whole finance team",
    "Everyone on the 3rd floor lost network connectivity",
]


# ---------------------------------------------------------------- loading ---
def load_articles(kb_dir: Path) -> list[dict]:
    """Read every .md file and return [{file, title, text}]."""
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")

    articles = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        articles.append({"file": path.name, "title": title, "text": text})
    return articles


# --------------------------------------------------------------- chunking ---
def split_at_h2(text: str) -> list[tuple[str, str]]:
    """
    Split markdown at level-2 headings ('## ').
    '###' sub-headings stay inside their parent '##' section.
    Text before the first '##' (title + metadata) becomes an 'Overview' chunk.
    Returns [(section_heading, section_text)].
    """
    parts = re.split(r"(?m)^(?=##\s)", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            heading = part.splitlines()[0][3:].strip()
        else:
            heading = "Overview"
        chunks.append((heading, part))
    return chunks


def build_chunks(articles: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    ids, documents, metadatas = [], [], []
    for art in articles:
        for idx, (section, body) in enumerate(split_at_h2(art["text"])):
            ids.append(f"{Path(art['file']).stem}::{idx}")
            # Prefix the article title so every chunk carries its context
            documents.append(f"{art['title']}\n\n{body}")
            metadatas.append({
                "article_file": art["file"],
                "article_title": art["title"],
                "section": section,
                "chunk_index": idx,
            })
    return ids, documents, metadatas


# ---------------------------------------------------------------- storing ---
def build_collection(ids, documents, metadatas):
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Rebuild from scratch on every run so re-runs don't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance -> easy 0..1 score
    )
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


# --------------------------------------------------------------- querying ---
def best_article(collection, query: str, n_results: int = 5) -> dict:
    """
    Retrieve the top chunks, then pick the article whose best chunk scores
    highest. Confidence = 1 - cosine distance (1.0 = identical meaning).
    """
    res = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas", "distances"],
    )
    best = None
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        confidence = max(0.0, 1.0 - dist)
        if best is None or confidence > best["confidence"]:
            best = {
                "title": meta["article_title"],
                "file": meta["article_file"],
                "section": meta["section"],
                "confidence": confidence,
            }
    return best


def main():
    articles = load_articles(KB_DIR)
    ids, documents, metadatas = build_chunks(articles)
    collection = build_collection(ids, documents, metadatas)

    print(f"Loaded {len(articles)} articles from {KB_DIR}")
    for art in articles:
        n = sum(1 for m in metadatas if m["article_file"] == art["file"])
        print(f"  - {art['file']:<28} {n} chunks")
    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'\n")

    print("=" * 78)
    print("Sample query results")
    print("=" * 78)
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        hit = best_article(collection, q)
        print(f"\n[{i}] Query      : {q}")
        print(f"    Article    : {hit['title']}  ({hit['file']})")
        print(f"    Section    : {hit['section']}")
        print(f"    Confidence : {hit['confidence']:.2%}")


if __name__ == "__main__":
    main()