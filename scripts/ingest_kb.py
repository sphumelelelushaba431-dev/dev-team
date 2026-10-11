"""Chunk, embed and index the documents in knowledge_base/ into the vector database."""
import _path  # noqa: F401
from app.rag.ingest import build_kb_index

if __name__ == "__main__":
    build_kb_index()
