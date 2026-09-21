from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import User, QueryLog
from app.rag.chain import answer_question
from app.schemas import QueryRequest, QueryResponse, RetrievedChunk

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Runs the full RAG pipeline scoped to the authenticated user's role and
    department. Every query is written to QueryLog for auditability.
    """
    result = answer_question(
        question=payload.question,
        role=user.role,
        department=user.department,
        k=payload.top_k,
    )

    retrieved = [
        RetrievedChunk(
            source=c["metadata"].get("source", "unknown"),
            department=c["metadata"].get("department", "unknown"),
            access_level=c["metadata"].get("access_level", "unknown"),
            content_preview=c["content"][:200],
            score=c.get("distance"),
        )
        for c in result["chunks"]
    ]

    log = QueryLog(
        user_id=user.id,
        question=payload.question,
        answer=result["answer"],
        retrieved_doc_ids=",".join(c["metadata"].get("doc_id", "") for c in result["chunks"]),
        latency_ms=result["latency_ms"],
    )
    db.add(log)
    db.commit()

    return QueryResponse(
        answer=result["answer"],
        retrieved_chunks=retrieved,
        latency_ms=result["latency_ms"],
    )
