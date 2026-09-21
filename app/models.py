import enum
import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Enum, ForeignKey, Text, Integer
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class RoleEnum(str, enum.Enum):
    """
    Role hierarchy used for RBAC filtering at retrieval time.
    Order matters: each role can see its own access_level and everything
    "below" it in sensitivity (see ROLE_ACCESS_MAP in app/rag/vectorstore.py).
    """
    ADMIN = "admin"          # sees confidential + internal + public, all departments
    MANAGER = "manager"      # sees internal + public, own department + public depts
    EMPLOYEE = "employee"    # sees public + internal for own department only
    GUEST = "guest"          # sees public only


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(RoleEnum), nullable=False, default=RoleEnum.EMPLOYEE)
    department = Column(String, nullable=False, default="general")
    is_active = Column(Integer, default=1)  # 1/0 instead of Boolean for cross-db portability
    created_at = Column(DateTime, default=datetime.utcnow)

    query_logs = relationship("QueryLog", back_populates="user")


class DocumentRecord(Base):
    """
    SQL-side bookkeeping for every document ingested into the vector store.
    ChromaDB holds the actual chunks + embeddings; this table is the
    system-of-record for auditing, access-level management, and re-ingestion.
    """
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    department = Column(String, nullable=False, default="general")
    access_level = Column(String, nullable=False, default="internal")  # public | internal | confidential
    chunk_count = Column(Integer, default=0)
    uploaded_by = Column(String, ForeignKey("users.id"), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="ingested")  # ingested | failed | pending


class QueryLog(Base):
    """Audit trail of every RAG query, for observability / compliance."""
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    retrieved_doc_ids = Column(Text, nullable=True)  # comma-separated chunk ids
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="query_logs")
