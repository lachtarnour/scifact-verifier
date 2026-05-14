"""Flask routes — chat with streaming SSE."""

import json
from pathlib import Path

from flask import Blueprint, Response, jsonify, render_template, request

from app.services import search, stream_generation
from src.utils import config, get_logger

logger = get_logger(__name__)
bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------
# Streaming endpoint (SSE)
# ------------------------------------------------------------------

@bp.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body: {"claim": "...", "mode": "hybrid_rerank"}

    Returns a Server-Sent Events stream:
      - event: retrieval results (JSON)
      - events: tokens (streamed one by one)
      - event: done
    """
    data = request.get_json(force=True, silent=True) or {}
    claim = data.get("claim", "").strip()
    mode = data.get("mode", "hybrid_rerank")
    model = data.get("model")

    if not claim:
        return jsonify({"error": "Missing 'claim' field."}), 400

    if len(claim) < 10:
        return jsonify({"error": "Claim is too short (min 10 characters)."}), 400

    try:
        return Response(
            stream_generation(claim, mode=mode, model=model),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error("Streaming error: %s", e)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# Non-streaming endpoint (retrieval only)
# ------------------------------------------------------------------

@bp.route("/api/search", methods=["POST"])
def api_search():
    """
    POST /api/search
    Body: {"claim": "...", "mode": "hybrid_rerank"}
    """
    data = request.get_json(force=True, silent=True) or {}
    claim = data.get("claim", "").strip()
    mode = data.get("mode", "hybrid_rerank")

    if not claim:
        return jsonify({"error": "Missing 'claim' field."}), 400

    try:
        result = search(claim, mode=mode)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.error("API error: %s", e)
        return jsonify({"error": "Internal server error."}), 500


@bp.route("/api/metrics")
def metrics():
    """Serve saved retrieval metrics (or empty dict)."""
    metrics_path = config.REPORTS_DIR / "retrieval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return jsonify(json.load(f))
    return jsonify({})


@bp.route("/api/health")
def health():
    from app.services import RetrievalService
    try:
        svc = RetrievalService.get()
        return jsonify({"status": "ok", "corpus_size": len(svc.corpus)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503
