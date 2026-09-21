from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.routers import auth_router, documents_router, query_router

# Create SQL tables on startup (use Alembic migrations in a real deployment)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Full-stack Retrieval-Augmented Generation service with JWT auth and "
        "role-based access control (RBAC) enforced at the vector-retrieval layer."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(documents_router.router)
app.include_router(query_router.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "env": settings.ENV, "llm_provider": settings.LLM_PROVIDER}
