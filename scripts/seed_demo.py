"""
One-shot demo setup:
  1. Creates the SQL tables.
  2. Seeds 4 demo users covering each role.
  3. Ingests the sample documents in data/sample_docs with realistic
     department + access_level tags.

Run:  python scripts/seed_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Base, engine, SessionLocal
from app.models import User, RoleEnum
from app.security import hash_password
from app.rag.ingestion import ingest_file

DEMO_USERS = [
    dict(username="alice_admin", email="alice@technova.example", password="Admin123!",
         role=RoleEnum.ADMIN, department="general"),
    dict(username="bob_manager", email="bob@technova.example", password="Manager123!",
         role=RoleEnum.MANAGER, department="engineering"),
    dict(username="carol_employee", email="carol@technova.example", password="Employee123!",
         role=RoleEnum.EMPLOYEE, department="hr"),
    dict(username="dave_guest", email="dave@technova.example", password="Guest123!",
         role=RoleEnum.GUEST, department="general"),
]

DOCS = [
    ("public_company_overview.txt", "general", "public"),
    ("engineering_internal_architecture.txt", "engineering", "internal"),
    ("finance_confidential_q3_report.txt", "finance", "confidential"),
    ("hr_internal_benefits.txt", "hr", "internal"),
]


def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    print("Seeding users...")
    for u in DEMO_USERS:
        if db.query(User).filter(User.username == u["username"]).first():
            print(f"  - {u['username']} already exists, skipping")
            continue
        user = User(
            username=u["username"],
            email=u["email"],
            hashed_password=hash_password(u["password"]),
            role=u["role"],
            department=u["department"],
        )
        db.add(user)
        print(f"  + {u['username']} ({u['role'].value}, dept={u['department']}) "
              f"password={u['password']}")
    db.commit()
    db.close()

    print("\nIngesting sample documents...")
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    for filename, department, access_level in DOCS:
        path = sample_dir / filename
        summary = ingest_file(path, department=department, access_level=access_level)
        print(f"  + {filename} -> dept={department}, access={access_level}, "
              f"chunks={summary['chunk_count']}")

    print("\nDone. Try logging in as any demo user (see README for curl examples).")


if __name__ == "__main__":
    main()
