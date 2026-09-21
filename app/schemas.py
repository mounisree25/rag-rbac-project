from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr

from app.models import RoleEnum


# ---------- Auth ----------
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: RoleEnum = RoleEnum.EMPLOYEE
    department: str = "general"


class UserOut(BaseModel):
    id: str
    username: str
    email: EmailStr
    role: RoleEnum
    department: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: RoleEnum
    department: str


# ---------- Documents ----------
class DocumentOut(BaseModel):
    id: str
    filename: str
    department: str
    access_level: str
    chunk_count: int
    status: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


# ---------- Query ----------
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class RetrievedChunk(BaseModel):
    source: str
    department: str
    access_level: str
    content_preview: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: int
