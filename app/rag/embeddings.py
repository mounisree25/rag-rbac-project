"""
Embedding layer for the RAG pipeline.

Production note (see README "Swapping in production-grade models"):
    In the deployed version of this project we used sentence-transformer /
    OpenAI embedding models. This repo ships with a fully self-contained,
    offline embedding function (`LocalHashingEmbeddings`) built on
    scikit-learn's HashingVectorizer so the whole pipeline runs end-to-end
    with zero external model downloads / API keys, which makes it trivial
    to demo, unit test, and run in CI. Swapping providers is a one-line
    change (see `get_embedding_function`).
"""
from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from app.config import settings


class LocalHashingEmbeddings:
    """
    A stateless, offline embedding function compatible with LangChain's
    Embeddings interface (embed_documents / embed_query) and with
    ChromaDB's EmbeddingFunction protocol (__call__).

    Uses the hashing trick (HashingVectorizer) + L2-normalized TF weighting,
    so no vocabulary needs to be fit/persisted -> safe for incremental,
    concurrent document ingestion.
    """

    def __init__(self, n_features: int = 2 ** 12):
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32).tolist()

    # LangChain Embeddings interface (used if we later swap in a LangChain
    # vectorstore/retriever wrapper instead of the raw chromadb client)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query_lc(self, text: str) -> List[float]:
        return self._embed([text])[0]

    # ChromaDB EmbeddingFunction protocol: __call__ handles both documents
    # and queries (input is always a list of strings). Chroma calls
    # embed_query(input=...) specifically for query-time embedding; we
    # reuse the same embedding space for both (see docstring above).
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self._embed(input)

    def name(self) -> str:
        return "local-hashing-embeddings"

    @staticmethod
    def build_from_config(config: dict):
        return LocalHashingEmbeddings(n_features=config.get("n_features", 2 ** 12))

    def get_config(self) -> dict:
        return {"n_features": self._vectorizer.n_features}


def get_embedding_function():
    """
    Central factory so the rest of the app never hard-codes an embedding
    provider. Extend this to branch on settings.EMBEDDING_PROVIDER for
    OpenAIEmbeddings / HuggingFaceEmbeddings in a real deployment.
    """
    return LocalHashingEmbeddings()
