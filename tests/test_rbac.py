"""
Tests focus on the thing that actually matters for this project: that RBAC
is correctly enforced at the retrieval layer, and that the API surfaces it
correctly (401/403 on unauthorized access).

Run: pytest -v
"""
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_DB = ROOT / "test_rbac.db"
TEST_CHROMA = ROOT / "test_chroma_db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB}"
os.environ["CHROMA_PERSIST_DIR"] = str(TEST_CHROMA)
os.environ["LLM_PROVIDER"] = "extractive"

from app.main import app  # noqa: E402
from app.models import RoleEnum  # noqa: E402
from app.rag.ingestion import ingest_file  # noqa: E402
from app.rag import vectorstore  # noqa: E402

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown(tmp_path_factory):
    # --- seed documents across departments/access levels ---
    docs_dir = tmp_path_factory.mktemp("docs")
    (docs_dir / "public.txt").write_text("The company was founded in 2018 in Austin.")
    (docs_dir / "finance.txt").write_text("Q3 revenue was 6.2 million dollars, confidential.")
    (docs_dir / "eng.txt").write_text("The backend runs on Kubernetes and Kafka.")

    ingest_file(docs_dir / "public.txt", department="general", access_level="public")
    ingest_file(docs_dir / "finance.txt", department="finance", access_level="confidential")
    ingest_file(docs_dir / "eng.txt", department="engineering", access_level="internal")

    # --- seed users via the API itself ---
    users = [
        ("admin1", "admin1@x.com", "Password123!", "admin", "general"),
        ("eng_manager", "engm@x.com", "Password123!", "manager", "engineering"),
        ("hr_employee", "hre@x.com", "Password123!", "employee", "hr"),
        ("guest1", "guest1@x.com", "Password123!", "guest", "general"),
    ]
    for username, email, password, role, dept in users:
        client.post("/auth/register", json={
            "username": username, "email": email, "password": password,
            "role": role, "department": dept,
        })

    yield

    if TEST_DB.exists():
        TEST_DB.unlink()
    if TEST_CHROMA.exists():
        shutil.rmtree(TEST_CHROMA)


def _login(username, password="Password123!"):
    resp = client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_admin_sees_confidential_finance_doc():
    token = _login("admin1")
    resp = client.post("/query/", json={"question": "What was Q3 revenue?"},
                        headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    sources = [c["source"] for c in resp.json()["retrieved_chunks"]]
    assert "finance.txt" in sources


def test_hr_employee_cannot_see_confidential_finance_doc():
    token = _login("hr_employee")
    resp = client.post("/query/", json={"question": "What was Q3 revenue?"},
                        headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    sources = [c["source"] for c in resp.json()["retrieved_chunks"]]
    assert "finance.txt" not in sources


def test_engineering_manager_cannot_see_other_departments_internal_doc():
    token = _login("eng_manager")
    resp = client.post("/query/", json={"question": "employee benefits and revenue"},
                        headers={"Authorization": f"Bearer {token}"})
    sources = [c["source"] for c in resp.json()["retrieved_chunks"]]
    assert "finance.txt" not in sources
    assert "eng.txt" in sources or "public.txt" in sources


def test_guest_only_sees_public_docs():
    token = _login("guest1")
    resp = client.post("/query/", json={"question": "anything about the company"},
                        headers={"Authorization": f"Bearer {token}"})
    sources = [c["source"] for c in resp.json()["retrieved_chunks"]]
    assert all(s == "public.txt" for s in sources)


def test_unauthenticated_query_is_rejected():
    resp = client.post("/query/", json={"question": "anything"})
    assert resp.status_code == 401


def test_guest_cannot_upload_documents():
    token = _login("guest1")
    resp = client.post(
        "/documents/upload",
        files={"file": ("x.txt", b"hello world", "text/plain")},
        data={"department": "general", "access_level": "public"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


def test_build_rbac_filter_shapes():
    # Admin: no department restriction
    f = vectorstore.build_rbac_filter(RoleEnum.ADMIN, "general")
    assert "$and" not in f

    # Employee: department-scoped
    f = vectorstore.build_rbac_filter(RoleEnum.EMPLOYEE, "hr")
    assert "$and" in f
