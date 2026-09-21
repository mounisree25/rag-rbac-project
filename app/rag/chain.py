"""
Assembles the retrieve -> augment -> generate chain.

Retrieval is RBAC-filtered (see app/rag/vectorstore.py) *before* anything
reaches the LLM, so the model is architecturally incapable of leaking a
chunk the requesting user isn't authorized to see.
"""
import time
from typing import List

from app.config import settings
from app.models import RoleEnum
from app.rag import vectorstore

PROMPT_TEMPLATE = """You are a helpful enterprise assistant. Answer the question using ONLY
the information provided in the context below. If the answer is not in the
context, say "I don't have that information in my knowledge base." Do not
use outside knowledge, and do not reveal information from documents that
are not shown to you below.

Context:
{context}

Question: {question}

Answer:"""


def _format_context(chunks: List[dict]) -> str:
    return "\n\n".join(c["content"] for c in chunks)


def _call_llm(prompt: str, context_chunks: List[dict], question: str) -> str:
    provider = settings.LLM_PROVIDER

    if provider == "openai" and settings.OPENAI_API_KEY:
        from openai import OpenAI  # imported lazily so it's optional
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    # --- Extractive fallback (no external LLM call) ---
    # Deterministic, offline, and CI-safe: returns the single most relevant
    # chunk with a note, so the whole pipeline is demonstrable without an
    # API key. Swap LLM_PROVIDER=openai (+ OPENAI_API_KEY) for real
    # generative answers.
    if not context_chunks:
        return "I don't have that information in my knowledge base."
    best = context_chunks[0]
    return (
        f"[extractive mode] Based on the most relevant passage in "
        f"'{best['metadata'].get('source')}':\n\n{best['content']}"
    )


def answer_question(question: str, role: RoleEnum, department: str, k: int = None) -> dict:
    start = time.perf_counter()

    chunks = vectorstore.similarity_search(question, role=role, department=department, k=k)
    context = _format_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = _call_llm(prompt, chunks, question)

    latency_ms = int((time.perf_counter() - start) * 1000)

    return {
        "answer": answer,
        "chunks": chunks,
        "latency_ms": latency_ms,
    }
