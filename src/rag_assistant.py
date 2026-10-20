"""Local document retrieval using TF-IDF with traceable source labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SUPPORTED = {".txt", ".md"}


@dataclass(frozen=True)
class TextChunk:
    source: str
    chunk_id: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    source_id: str
    source: str
    chunk_id: int
    score: float
    text: str


def chunk_text(text: str, chunk_size: int = 850, overlap: int = 120) -> list[str]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use a positive chunk size and overlap between zero and chunk_size.")
    cleaned = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start + int(chunk_size * 0.65), end)
            if boundary > start:
                end = boundary
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_chunks(directory: Path, chunk_size: int = 850, overlap: int = 120) -> list[TextChunk]:
    chunks = []
    if not directory.exists():
        return chunks
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            text = path.read_text(encoding="utf-8", errors="replace")
            for idx, piece in enumerate(chunk_text(text, chunk_size, overlap), start=1):
                chunks.append(TextChunk(path.relative_to(directory).as_posix(), idx, piece))
    return chunks


class ResearchAssistant:
    def __init__(self, documents_dir: Path, chunk_size: int = 850, overlap: int = 120):
        self.documents_dir = Path(documents_dir)
        self.chunks = load_chunks(self.documents_dir, chunk_size, overlap)
        self.vectorizer = None
        self.matrix = None
        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not query.strip() or self.matrix is None or self.vectorizer is None:
            return []
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        ranked = np.argsort(scores)[::-1]
        found = []
        for index in ranked:
            if scores[index] <= 0:
                continue
            chunk = self.chunks[int(index)]
            found.append(SearchResult(f"S{len(found) + 1}", chunk.source, chunk.chunk_id, float(scores[index]), chunk.text))
            if len(found) >= top_k:
                break
        return found

    def answer(self, query: str, top_k: int = 4) -> dict[str, object]:
        results = self.retrieve(query, top_k)
        if not results:
            answer = "No matching passages were found in the indexed documents."
        else:
            answer = "\n\n".join(f"{item.text} [{item.source_id}]" for item in results[:3])
            answer += "\n\nCheck the source passages before relying on this summary."
        return {"answer": answer, "sources": results, "mode": "local TF-IDF retrieval"}
