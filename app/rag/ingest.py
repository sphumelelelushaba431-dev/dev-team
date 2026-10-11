"""Knowledge base ingestion: load documents, chunk, embed and index.

Supports .md, .txt and .pdf files in knowledge_base/. Each file's name
(e.g. finance_policy.md) is used to tag its chunks with a department.
"""
from __future__ import annotations

import re
from pathlib import Path

from .. import config
from .embedder import make_embedder, save_embedder
from .vector_store import VectorStore

DEPT_HINTS = {"academic": "Academic", "finance": "Finance", "ict": "ICT",
              "facilit": "Facilities", "hostel": "Hostel", "residence": "Hostel"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("Install pypdf to ingest PDF files") from e
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, words: int = None, overlap: int = None):
    """Split by section headings first, then into overlapping word windows."""
    words = words or config.CHUNK_WORDS
    overlap = overlap or config.CHUNK_OVERLAP
    sections = re.split(r"\n(?=#+ )", text)
    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else ""
        tokens = " ".join(lines[1:] if heading else lines).split()
        if not tokens:
            continue
        step = max(1, words - overlap)
        for start in range(0, len(tokens), step):
            piece = " ".join(tokens[start:start + words])
            yield heading, piece
            if start + words >= len(tokens):
                break


def department_for(name: str) -> str:
    low = name.lower()
    for key, dept in DEPT_HINTS.items():
        if key in low:
            return dept
    return "General"


def build_kb_index(kb_dir: Path = None, out_prefix: Path = None, verbose: bool = True):
    kb_dir = Path(kb_dir or config.KB_DIR)
    out_prefix = Path(out_prefix or config.ARTIFACTS_DIR / "kb")
    chunks = []
    files = sorted(p for p in kb_dir.iterdir() if p.suffix.lower() in (".md", ".txt", ".pdf"))
    for path in files:
        dept = department_for(path.stem)
        for i, (heading, piece) in enumerate(chunk_text(read_document(path))):
            text = f"{heading}. {piece}" if heading else piece
            chunks.append({"id": f"{path.stem}#{i}", "source": path.name, "section": heading,
                           "department": dept, "text": text})
    if not chunks:
        raise RuntimeError(f"No documents found in {kb_dir}")
    embedder = make_embedder().fit([c["text"] for c in chunks])
    vectors = embedder.encode([c["text"] for c in chunks])
    store = VectorStore(vectors.shape[1])
    store.add(vectors, chunks)
    store.save(out_prefix)
    save_embedder(embedder, out_prefix)
    if verbose:
        print(f"Indexed {len(chunks)} chunks from {len(files)} documents "
              f"({embedder.kind} embeddings, {store.backend} index)")
    return store, embedder
