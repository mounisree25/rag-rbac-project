"""
Central configuration for the RAG-RBAC service.
All values are overridable via environment variables / a .env file.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "RAG Chatbot with Role-Based Access Control"
    ENV: str = os.getenv("ENV", "development")

    # --- Security / JWT ---
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- Database (SQL: user/auth store) ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./rbac_rag.db")

    # --- Vector store (ChromaDB) ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION: str = "enterprise_docs"

    # --- RAG pipeline ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 4

    # --- LLM provider ---
    # "openai"  -> real generation via OpenAI-compatible API (needs OPENAI_API_KEY)
    # "extractive" -> no external LLM call; returns the best-matching context
    #                 (used as an offline / CI-safe fallback, and to keep the
    #                 pipeline fully demonstrable without paid API access)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "extractive")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # --- Ingestion webhook (Flask microservice, triggered by n8n) ---
    INGEST_WEBHOOK_TOKEN: str = os.getenv("INGEST_WEBHOOK_TOKEN", "CHANGE_ME_WEBHOOK_TOKEN")

    class Config:
        env_file = ".env"


settings = Settings()
