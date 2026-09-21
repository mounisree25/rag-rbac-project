"""
Ingestion pipeline: text -> chunks -> embeddings -> ChromaDB.

Supports .txt / .md natively and .pdf via pypdf when available. Each chunk
is stamped with `department` and `access_level` metadata so the retrieval
layer (app/rag/vectorstore.py) can enforce RBAC later.
"""
import hashlib
import uuid
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag import vectorstore

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    length_function=len,
)


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("pypdf is required to ingest PDF files (pip install pypdf)") from e
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str) -> List[str]:
    return _splitter.split_text(text)


def ingest_file(path: Path, department: str, access_level: str) -> dict:
    """
    Loads a single file, splits it into chunks, embeds them, and writes them
    into ChromaDB with RBAC metadata. Returns a summary dict for logging /
    the SQL DocumentRecord.
    """
    if access_level not in vectorstore.ACCESS_LEVELS:
        raise ValueError(f"access_level must be one of {vectorstore.ACCESS_LEVELS}")

    raw_text = _read_text(path)
    chunks = chunk_text(raw_text)
    if not chunks:
        return {"filename": path.name, "chunk_count": 0}

    doc_id = str(uuid.uuid4())
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": path.name,
            "department": department,
            "access_level": access_level,
            "doc_id": doc_id,
            "chunk_index": i,
            "checksum": hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:12],
        }
        for i, chunk in enumerate(chunks)
    ]

    vectorstore.add_chunks(ids=ids, texts=chunks, metadatas=metadatas)

    return {
        "doc_id": doc_id,
        "filename": path.name,
        "department": department,
        "access_level": access_level,
        "chunk_count": len(chunks),
    }


def ingest_directory(directory: Path, department: str, access_level: str) -> List[dict]:
    summaries = []
    for path in sorted(directory.glob("**/*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".md", ".pdf"):
            summaries.append(ingest_file(path, department, access_level))
    return summaries
