"""MOTHER chat web app.

A simple page where Darius can talk to MOTHER with no code.
Each answer gets a thumbs up / thumbs down plus a "why" field;
the feedback is stored in SQLite and exported for the next
training round.

Configuration (environment variables):
  MOTHER_MODEL_PATH     weights file, default models/mother_v008_s11.npz
  MOTHER_TOKENIZER_PATH SentencePiece .model file, default models/mother_v006.model
  MOTHER_MAX_TOKENS     max new tokens per answer, default 48
  FEEDBACK_DB           SQLite file, default feedback.db (in the app dir)

The model is loaded ONCE at startup. If the weights or tokenizer
file is missing, the app still starts (so the feedback pages keep
working) but the chat shows a clear error naming the missing file.
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import sentencepiece as spm
from flask import Flask, g, jsonify, render_template, request, Response

from mother_model import BOS, EOS, PAD, generate, load_champion

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get(
    "MOTHER_MODEL_PATH", BASE_DIR / "models" / "mother_v008_s11.npz"))
TOKENIZER_PATH = Path(os.environ.get(
    "MOTHER_TOKENIZER_PATH", BASE_DIR / "models" / "mother_v006.model"))
MAX_TOKENS = int(os.environ.get("MOTHER_MAX_TOKENS", "48"))
DB_PATH = Path(os.environ.get("FEEDBACK_DB", BASE_DIR / "feedback.db"))

app = Flask(__name__)

# The model + tokenizer, loaded once. A lock guards generation so
# concurrent requests are answered one at a time (the model is
# small and CPU inference is fast; this keeps memory flat).
_model = None
_sp = None
_model_info = None
_model_error = None
_gen_lock = threading.Lock()


def _missing(what, path):
    return (f"MOTHER is not ready: the {what} file is missing: {path}. "
            f"Put the file there (see README.md) and restart the service.")


def load_model():
    """Load weights + tokenizer once. Returns an error string, or None."""
    global _model, _sp, _model_info, _model_error
    if _model is not None:
        return None
    if _model_error is not None and _model is None:
        # A previous attempt failed; retry (the file may have arrived).
        _model_error = None
    if not MODEL_PATH.is_file():
        _model_error = _missing("weights", MODEL_PATH)
        return _model_error
    if not TOKENIZER_PATH.is_file():
        _model_error = _missing("tokenizer", TOKENIZER_PATH)
        return _model_error
    try:
        model, info = load_champion(str(MODEL_PATH))
        sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    except Exception as exc:  # noqa: BLE001 - surfaced to the user plainly
        _model_error = f"MOTHER failed to load: {exc}"
        return _model_error
    if sp.get_piece_size() != 8000:
        _model_error = (
            f"MOTHER failed to load: tokenizer vocabulary is "
            f"{sp.get_piece_size()}, expected 8000. Wrong tokenizer file?")
        return _model_error
    if (sp.bos_id(), sp.eos_id(), sp.pad_id()) != (BOS, EOS, PAD):
        _model_error = (
            "MOTHER failed to load: tokenizer special-token ids do not "
            f"match BOS={BOS} EOS={EOS} PAD={PAD}. Wrong tokenizer file?")
        return _model_error
    _model = model
    _sp = sp
    _model_info = info
    return None


# Try once at startup so a missing file shows up in the logs early.
load_model()


# ---------------------------------------------------------------- db ----
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute(
            """CREATE TABLE IF NOT EXISTS exchanges (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   question TEXT NOT NULL,
                   answer TEXT NOT NULL,
                   rating TEXT,
                   reason TEXT,
                   created_at TEXT NOT NULL,
                   rated_at TEXT
               )"""
        )
        g.db.commit()
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -------------------------------------------------------------- routes --
@app.route("/")
def index():
    return render_template(
        "index.html",
        model_ready=_model is not None,
        model_error=_model_error,
        model_info=_model_info,
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": _model is not None,
        "model_error": _model_error,
        "model": _model_info,
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Type a question first."}), 400
    if len(question) > 500:
        return jsonify({"error": "Please keep questions under 500 characters."}), 400
    err = load_model()
    if err is not None:
        return jsonify({"error": err}), 503
    with _gen_lock:
        t0 = time.time()
        try:
            answer = generate(_model, _sp, question, max_new_tokens=MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001 - generation must never 500 as HTML
            return jsonify({
                "error": f"MOTHER hit a problem answering that: {exc}. "
                         f"Try asking again."}), 500
        elapsed = round(time.time() - t0, 2)
    db = get_db()
    cur = db.execute(
        "INSERT INTO exchanges (question, answer, created_at) "
        "VALUES (?, ?, ?)",
        (question, answer, _now()),
    )
    db.commit()
    return jsonify({
        "exchange_id": cur.lastrowid,
        "question": question,
        "answer": answer,
        "seconds": elapsed,
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    try:
        exchange_id = int(data.get("exchange_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "Missing exchange id."}), 400
    rating = (data.get("rating") or "").strip().lower()
    if rating not in ("up", "down"):
        return jsonify({"error": "Rating must be 'up' or 'down'."}), 400
    reason = (data.get("reason") or "").strip()[:1000]
    db = get_db()
    cur = db.execute(
        "UPDATE exchanges SET rating = ?, reason = ?, rated_at = ? "
        "WHERE id = ?",
        (rating, reason, _now(), exchange_id),
    )
    db.commit()
    if cur.rowcount == 0:
        return jsonify({"error": "No such answer."}), 404
    return jsonify({"ok": True})


@app.route("/feedback", methods=["GET"])
def feedback_list():
    db = get_db()
    rows = db.execute(
        "SELECT id, question, answer, rating, reason, created_at, rated_at "
        "FROM exchanges ORDER BY id DESC LIMIT 200"
    ).fetchall()
    return render_template("feedback.html", rows=rows)


@app.route("/feedback/export")
def feedback_export():
    fmt = (request.args.get("format") or "json").lower()
    db = get_db()
    rows = db.execute(
        "SELECT id, question, answer, rating, reason, created_at, rated_at "
        "FROM exchanges ORDER BY id ASC"
    ).fetchall()
    items = [dict(r) for r in rows]
    if fmt == "csv":
        import csv
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=[
            "id", "question", "answer", "rating", "reason",
            "created_at", "rated_at"])
        w.writeheader()
        w.writerows(items)
        return Response(
            buf.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition":
                     "attachment; filename=mother-feedback.csv"})
    return Response(
        json.dumps(items, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition":
                 "attachment; filename=mother-feedback.json"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
