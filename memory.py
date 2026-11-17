"""Conversation memory for MOTHER.

MOTHER's chat page remembers what was said earlier in the same
browser session, so she can answer follow-up questions instead of
treating every question as the first one.

How it works
------------
* Each browser gets a random session id, carried in Flask's signed
  session cookie (see app.py). The id is just a key; nothing about
  the conversation lives in the cookie itself.
* Every exchange is stored in the feedback SQLite database with its
  session id (the ``session_id`` column on ``exchanges``).
* Before answering, /ask loads the last few exchanges for this
  session and folds them into the question string handed to
  generate(), in this shape::

      Conversation:
      User: {previous question}
      MOTHER: {previous answer}
      User: {current question}

  generate() wraps that in ``Lesson: ...\\nExplanation:`` as usual,
  so no change to mother_model.py is needed. When a web search also
  applies, a ``Context: {...}`` line is appended after the
  current-question line (history first, then the question, then the
  search context). When there is no history, the prompt is exactly
  as it was before memory existed.

* The question stored in the database stays the ORIGINAL question.
  History and search context never leak into stored data or the
  feedback export, so they never become training data by accident.

Limits
------
* History is capped at 4 exchanges and ~800 characters, so the
  prompt stays well under her 512-token context window.
* Memory lives in the same SQLite database as the feedback, which
  is ephemeral on Render's free tier: a redeploy or restart wipes
  it. Durable storage is a future decision, not part of this change.
* MOTHER was never trained on multi-turn dialogue, so memory is
  best-effort until a training round teaches her the Conversation
  shape. The shape in this file is the contract the lessons are
  written to, so do not change it without changing the lessons.
"""

import secrets

# Caps for the history folded into the prompt. 4 exchanges at ~200
# chars each is ~800 chars; with the current question (<= 500) and a
# search context (<= 600), the whole prompt stays around 475 tokens
# at most, inside her 512-token window.
MAX_EXCHANGES = 4
MAX_CHARS = 800


def new_session_id():
    """A random 128-bit session id, hex-encoded."""
    return secrets.token_hex(16)


def cap_history(history, max_exchanges=MAX_EXCHANGES, max_chars=MAX_CHARS):
    """Trim an oldest-first [(question, answer), ...] list to the cap.

    Keeps the MOST RECENT exchanges that fit: at most max_exchanges
    of them, and at most max_chars of formatted history text. The
    most recent exchange is always kept, even if it alone is over
    the character budget, so memory never silently empties on a
    long last answer. Returns the kept list, still oldest-first.
    """
    kept = []
    total = len("Conversation:\n")
    for q, a in reversed(history):
        if len(kept) >= max_exchanges:
            break
        chunk = f"User: {q}\nMOTHER: {a}\n"
        if kept and total + len(chunk) > max_chars:
            break
        kept.append((q, a))
        total += len(chunk)
    kept.reverse()
    return kept


def build_memory_question(question, history, context=""):
    """Fold history (+ optional search context) into the question string.

    ``history`` is an oldest-first [(question, answer), ...] list,
    already capped by cap_history(). The result is the string /ask
    hands to generate()::

        Conversation:
        User: {previous question}
        MOTHER: {previous answer}
        User: {current question}
        Context: {search results}      (only when context is given)

    With no history this is just the question (plus the Context
    line when one is given), exactly what build_grounded_question
    produces — the prompt is unchanged from before memory existed.
    """
    if not history:
        if context:
            return f"{question}\nContext: {context}"
        return question
    lines = ["Conversation:"]
    for q, a in history:
        lines.append(f"User: {q}")
        lines.append(f"MOTHER: {a}")
    lines.append(f"User: {question}")
    if context:
        lines.append(f"Context: {context}")
    return "\n".join(lines)
