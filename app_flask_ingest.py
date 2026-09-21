"""
Lightweight Flask microservice that exposes a single webhook used by n8n
(see n8n/document_ingestion_workflow.json) to trigger automated document
ingestion whenever a new file lands in a watched folder / cloud bucket.

Why a separate Flask service instead of adding this to the FastAPI app?
  - Keeps the ingestion path decoupled from the user-facing query API, so a
    slow/bulk ingestion job can't add latency or contention to live chat
    traffic (separate process, can be scaled/deployed independently).
  - Matches how it was actually deployed: FastAPI = synchronous, low-latency
    query API; Flask = simple internal automation endpoint fronted by n8n.

Run with:  flask --app app_flask_ingest run --port 5001
"""
import os
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify

from app.config import settings
from app.rag.ingestion import ingest_file
from app.database import SessionLocal
from app.models import DocumentRecord

app = Flask(__name__)


def _authorized(req) -> bool:
    token = req.headers.get("X-Webhook-Token", "")
    return token == settings.INGEST_WEBHOOK_TOKEN


@app.route("/webhooks/ingest", methods=["POST"])
def ingest_webhook():
    """
    Expected multipart/form-data payload (this is exactly what the n8n
    'HTTP Request' node sends after its Trigger + Function nodes classify
    the file's department/access_level):
        file: <binary>
        department: str
        access_level: public|internal|confidential
    """
    if not _authorized(request):
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    department = request.form.get("department", "general")
    access_level = request.form.get("access_level", "internal")
    uploaded = request.files["file"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / uploaded.filename
        uploaded.save(tmp_path)
        try:
            summary = ingest_file(tmp_path, department=department, access_level=access_level)
        except Exception as e:
            return jsonify({"error": str(e)}), 422

    db = SessionLocal()
    try:
        record = DocumentRecord(
            filename=summary["filename"],
            department=department,
            access_level=access_level,
            chunk_count=summary.get("chunk_count", 0),
            status="ingested",
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

    return jsonify(summary), 201


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=int(os.getenv("FLASK_PORT", 5001)))
