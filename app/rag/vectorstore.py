"""
Vector store access layer.

This is where Role-Based Access Control actually gets enforced for
retrieval: every chunk stored in ChromaDB carries `department` and
`access_level` metadata at ingestion time, and every query builds a
Chroma `where` filter derived from the *authenticated* user's role and
department before the similarity search runs. A user therefore can never
retrieve (and the LLM can never see) a chunk it isn't entitled to -
access control happens at the retrieval layer, not just at the API layer.
"""
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.models import RoleEnum
from app.rag.embeddings import get_embedding_function

_client = chromadb.PersistentClient(
    path=settings.CHROMA_PERSIST_DIR,
    settings=ChromaSettings(anonymized_telemetry=False),
)

_embedding_fn = get_embedding_function()

ACCESS_LEVELS = ["public", "internal", "confidential"]

# Which access levels each role is allowed to see at all.
ROLE_ACCESS_LEVELS = {
    RoleEnum.ADMIN: ["public", "internal", "confidential"],
    RoleEnum.MANAGER: ["public", "internal", "confidential"],
    RoleEnum.EMPLOYEE: ["public", "internal"],
    RoleEnum.GUEST: ["public"],
}

# Whether the role is restricted to its own department for non-public docs.
ROLE_DEPARTMENT_SCOPED = {
    RoleEnum.ADMIN: False,
    RoleEnum.MANAGER: True,
    RoleEnum.EMPLOYEE: True,
    RoleEnum.GUEST: False,
}


def get_collection():
    return _client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_rbac_filter(role: RoleEnum, department: str) -> dict:
    """
    Build a ChromaDB `where` clause that restricts retrieval to chunks the
    given (role, department) is authorized to read.
    """
    allowed_levels = ROLE_ACCESS_LEVELS[role]
    level_clause = {"access_level": {"$in": allowed_levels}}

    if not ROLE_DEPARTMENT_SCOPED[role]:
        # Admin / guest: no department restriction beyond access level.
        return level_clause

    # Department-scoped roles: allowed if it's public OR it's their own department.
    dept_clause = {
        "$or": [
            {"access_level": "public"},
            {"department": department},
        ]
    }
    return {"$and": [level_clause, dept_clause]}


def add_chunks(
    ids: List[str],
    texts: List[str],
    metadatas: List[dict],
) -> None:
    collection = get_collection()
    collection.add(ids=ids, documents=texts, metadatas=metadatas)


def similarity_search(
    query: str,
    role: RoleEnum,
    department: str,
    k: Optional[int] = None,
) -> List[dict]:
    """
    Run an RBAC-filtered similarity search and return the top-k chunks with
    their metadata and distance score.
    """
    collection = get_collection()
    where = build_rbac_filter(role, department)
    n_results = k or settings.TOP_K

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
    )

    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({"content": doc, "metadata": meta, "distance": dist})
    return chunks
