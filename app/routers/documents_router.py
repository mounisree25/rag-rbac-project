import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_user, require_roles
from app.database import get_db
from app.models import User, DocumentRecord, RoleEnum
from app.rag.ingestion import ingest_file
from app.rag.vectorstore import ACCESS_LEVELS
from app.schemas import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentOut,
    dependencies=[Depends(require_roles(RoleEnum.ADMIN, RoleEnum.MANAGER))],
)
def upload_document(
    file: UploadFile = File(...),
    department: str = Form(...),
    access_level: str = Form("internal"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Only admins/managers can ingest documents. Managers may only tag
    documents into their own department; admins can tag any department.
    """
    if access_level not in ACCESS_LEVELS:
        raise HTTPException(400, f"access_level must be one of {ACCESS_LEVELS}")
    if user.role == RoleEnum.MANAGER and department != user.department:
        raise HTTPException(403, "Managers may only upload documents to their own department")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / file.filename
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        summary = ingest_file(tmp_path, department=department, access_level=access_level)

    record = DocumentRecord(
        filename=summary["filename"],
        department=department,
        access_level=access_level,
        chunk_count=summary.get("chunk_count", 0),
        uploaded_by=user.id,
        status="ingested" if summary.get("chunk_count", 0) > 0 else "failed",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/", response_model=List[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(RoleEnum.ADMIN, RoleEnum.MANAGER)),
):
    return db.query(DocumentRecord).order_by(DocumentRecord.uploaded_at.desc()).all()
