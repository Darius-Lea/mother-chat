"""Tests for the MOTHER conversation-memory feature.

Offline and deterministic: no network calls. Run from the repo root
with a Python that has flask + sentencepiece installed:

    python tests/test_memory.py

Covers:
  * memory.py: session ids, the Conversation prompt shape, the
    4-exchange / ~800-char caps, search-context ordering
  * the full Flask /ask flow with a stubbed model: history is
    included on the second question of a session, not for a fresh
    session, capped at 4 exchanges, stored questions stay original,
    search + memory combine in the right order, memory disabled
    returns remembered=0, a history lookup failure still answers,
    feedback + export still work, and a pre-memory database is
    migrated in place.
"""

import os
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ setup
# The real mother_model needs torch (not on this machine), so stub it
# before importing app. The stub records what generate() is handed.
GENERATE_CALLS = []


def _fake_generate(model, sp, question, max_new_tokens=48):
    GENERATE_CALLS.append(question)
    return "STUB ANSWER"


fake_mm = types.ModuleType("mother_model")
fake_mm.BOS, fake_mm.EOS, fake_mm.PAD = 2, 3, 0
fake_mm.generate = _fake_generate
fake_mm.load_champion = lambda p: (object(), {
    "stage": "STUB", "epoch": 0, "params": 1})
sys.modules["mother_model"] = fake_mm

_tmp = tempfile.TemporaryDirectory()
_dummy_weights = Path(_tmp.name) / "dummy.pt"
_dummy_weights.write_bytes(b"dummy")
_dummy_tok = Path(_tmp.name) / "dummy.model"
_dummy_tok.write_bytes(b"dummy")
os.environ["MOTHER_MODEL_PATH"] = str(_dummy_weights)
os.environ["MOTHER_TOKENIZER_PATH"] = str(_dummy_tok)
DB_PATH = Path(_tmp.name) / "test-feedback.db"
os.environ["FEEDBACK_DB"] = str(DB_PATH)

import memory  # noqa: E402
import app as appmod  # noqa: E402

# Pretend the model loaded (load_model failed on the dummy tokenizer,
# which is expected; the stub stands in for the real model).
appmod._model = object()
appmod._sp = object()
appmod._model_info = {"stage": "STUB", "epoch": 0, "params": 1}
appmod._model_error = None
appmod.generate = _fake_generate

PASSED = 0
FAILED = 0


def check(name, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  {name}")
    else:
        FAILED += 1
        print(f"FAIL  {name}  {extra}")


# --------------------------------------------------------- session ids ---
print("session ids")
sid1, sid2 = memory.new_session_id(), memory.new_session_id()
check("session id is 32 hex chars",
      len(sid1) == 32 and all(c in "0123456789abcdef" for c in sid1))
check("two session ids differ", sid1 != sid2)

# ------------------------------------------------------ prompt shape ----
print("prompt shape")
hist = [("What is a saxophone?", "A saxophone is a musical instrument."),
        ("What is a falcon?", "A falcon is a bird.")]
built = memory.build_memory_question("What is a platypus?", hist)
check("Conversation shape is exact",
      built == ("Conversation:\n"
                "User: What is a saxophone?\n"
                "MOTHER: A saxophone is a musical instrument.\n"
                "User: What is a falcon?\n"
                "MOTHER: A falcon is a bird.\n"
                "User: What is a platypus?"),
      f"got {built!r}")
built_ctx = memory.build_memory_question(
    "What is the news today?", hist, "Paris - Wikipedia: Paris is the capital of France.")
check("Context line goes after the current question",
      built_ctx.endswith("User: What is the news today?\n"
                         "Context: Paris - Wikipedia: Paris is the capital of France.")
      and built_ctx.startswith("Conversation:\n"),
      f"got {built_ctx!r}")
check("no history, no context -> question unchanged",
      memory.build_memory_question("Hi", []) == "Hi")
check("no history + context -> question + Context line",
      memory.build_memory_question("Hi", [], "some context")
      == "Hi\nContext: some context")
prompt = f"Lesson: {built}\nExplanation:"
check("prompt keeps the verified shape",
      prompt.startswith("Lesson: Conversation:\n")
      and prompt.endswith("Explanation:"))

# ------------------------------------------------------------- caps ------
print("caps")
six = [(f"Q{i}", f"A{i}") for i in range(1, 7)]
capped = memory.cap_history(six)
check("6 exchanges cap to 4",
      len(capped) == 4, f"got {len(capped)}")
check("cap keeps the most recent 4, oldest-first",
      capped == [(f"Q{i}", f"A{i}") for i in range(3, 7)],
      f"got {capped}")
long_hist = [("Q", "A" * 300) for _ in range(3)]
capped_long = memory.cap_history(long_hist)
check("char cap keeps the most recent that fit",
      len(capped_long) == 2, f"got {len(capped_long)}")
huge = [("Q", "A" * 900)]
check("most recent exchange is always kept, even over budget",
      memory.cap_history(huge) == huge)
check("empty history caps to []", memory.cap_history([]) == [])

# -------------------------------------------------------- Flask /ask -----
print("Flask /ask flow (stubbed model)")
client = appmod.app.test_client()

r = client.get("/health")
check("/health 200 + memory_enabled",
      r.status_code == 200 and r.get_json().get("memory_enabled") is True)

GENERATE_CALLS.clear()
r = client.post("/ask", json={"question": "What is a saxophone?"})
data = r.get_json()
check("first question in a session -> remembered=0",
      r.status_code == 200 and data.get("remembered") == 0, f"got {data}")
check("first question -> generate got the plain question",
      GENERATE_CALLS and GENERATE_CALLS[0] == "What is a saxophone?",
      f"got {GENERATE_CALLS[0][:100] if GENERATE_CALLS else 'none'}")
ex1 = data.get("exchange_id")

GENERATE_CALLS.clear()
r = client.post("/ask", json={"question": "What is a falcon?"})
data = r.get_json()
check("second question -> remembered=1",
      r.status_code == 200 and data.get("remembered") == 1, f"got {data}")
check("second question -> generate got the Conversation block",
      GENERATE_CALLS and GENERATE_CALLS[0] == (
          "Conversation:\n"
          "User: What is a saxophone?\n"
          "MOTHER: STUB ANSWER\n"
          "User: What is a falcon?"),
      f"got {GENERATE_CALLS[0][:160] if GENERATE_CALLS else 'none'}")
ex2 = data.get("exchange_id")

GENERATE_CALLS.clear()
r = client.post("/ask", json={"question": "What is 312 + 4?"})
data = r.get_json()
check("third question -> remembered=2",
      r.status_code == 200 and data.get("remembered") == 2, f"got {data}")
third_prompt = GENERATE_CALLS[0] if GENERATE_CALLS else ""
check("third question -> both prior exchanges, in order",
      third_prompt.index("User: What is a saxophone?")
      < third_prompt.index("User: What is a falcon?")
      < third_prompt.index("User: What is 312 + 4?"),
      f"got {third_prompt[:200]}")

# A fresh browser is a fresh conversation: no history leaks across.
other = appmod.app.test_client()
GENERATE_CALLS.clear()
r = other.post("/ask", json={"question": "What is a falcon?"})
data = r.get_json()
check("fresh session -> remembered=0",
      r.status_code == 200 and data.get("remembered") == 0, f"got {data}")
check("fresh session -> generate got the plain question",
      GENERATE_CALLS and GENERATE_CALLS[0] == "What is a falcon?")

# The 4-exchange cap, through the real flow.
cap_client = appmod.app.test_client()
for i in range(1, 7):
    GENERATE_CALLS.clear()
    r = cap_client.post("/ask", json={"question": f"Cap question {i}"})
    data = r.get_json()
sixth_prompt = GENERATE_CALLS[0] if GENERATE_CALLS else ""
check("6th question in a session -> remembered=4",
      data.get("remembered") == 4, f"got {data.get('remembered')}")
check("cap drops the oldest exchange from the prompt",
      "Cap question 1" not in sixth_prompt)
check("cap keeps the newest exchanges in the prompt",
      "Cap question 5" in sixth_prompt
      and sixth_prompt.count("User: ") == 5)  # 4 history + current

# The stored question stays the original: no history in the export.
r = client.get("/feedback/export")
export_text = r.get_data(as_text=True)
check("export has the plain questions",
      "What is a saxophone?" in export_text
      and "Cap question 3" in export_text)
check("export has no Conversation block leaked into storage",
      "Conversation:" not in export_text)

# -------------------------------------------------- search + memory -----
print("search + memory")
sample = [
    {"title": "Paris - Wikipedia",
     "snippet": "Paris is the capital of France.",
     "url": "https://en.wikipedia.org/wiki/Paris"},
]
real_search = appmod.search_web


def canned_search(question, max_results=3, timeout=4):
    return sample


appmod.search_web = canned_search
try:
    combo = appmod.app.test_client()
    combo.post("/ask", json={"question": "What is a saxophone?"})
    GENERATE_CALLS.clear()
    r = combo.post("/ask", json={"question": "What is the news today?"})
    data = r.get_json()
    combo_prompt = GENERATE_CALLS[0] if GENERATE_CALLS else ""
    check("trigger question with history -> searched=true",
          r.status_code == 200 and data.get("searched") is True,
          f"got {data}")
    check("trigger question with history -> remembered=1",
          data.get("remembered") == 1, f"got {data}")
    check("history, then question, then Context, in that order",
          combo_prompt.index("Conversation:")
          < combo_prompt.index("User: What is the news today?")
          < combo_prompt.index("Context: Paris - Wikipedia:"),
          f"got {combo_prompt[:220]}")
finally:
    appmod.search_web = real_search

# --------------------------------------------------- memory disabled ----
print("memory disabled")
appmod.MEMORY_ENABLED = False
try:
    off = appmod.app.test_client()
    off.post("/ask", json={"question": "What is a saxophone?"})
    GENERATE_CALLS.clear()
    r = off.post("/ask", json={"question": "What is a falcon?"})
    data = r.get_json()
    check("disabled -> remembered=0 on the second question",
          r.status_code == 200 and data.get("remembered") == 0,
          f"got {data}")
    check("disabled -> generate got the plain question",
          GENERATE_CALLS and GENERATE_CALLS[0] == "What is a falcon?")
    r = off.get("/health")
    check("disabled -> /health reports memory_enabled=false",
          r.get_json().get("memory_enabled") is False)
finally:
    appmod.MEMORY_ENABLED = True

# ------------------------------------------------- history lookup fails --
print("history lookup failure")
real_load = appmod._load_history


def broken_load(sid):
    raise RuntimeError("database locked")


appmod._load_history = broken_load
try:
    GENERATE_CALLS.clear()
    fail_client = appmod.app.test_client()
    r = fail_client.post("/ask", json={"question": "What is a saxophone?"})
    data = r.get_json()
    check("history lookup raising -> still 200 with an answer",
          r.status_code == 200 and data.get("answer") == "STUB ANSWER",
          f"got {data}")
    check("history lookup raising -> remembered=0, plain question",
          data.get("remembered") == 0
          and GENERATE_CALLS
          and GENERATE_CALLS[0] == "What is a saxophone?")
finally:
    appmod._load_history = real_load

# -------------------------------------------- feedback + export + migrate
print("feedback, export, migration")
r = client.post("/feedback", json={
    "exchange_id": ex2, "rating": "down", "reason": "should say more"})
check("feedback POST still works",
      r.status_code == 200 and r.get_json().get("ok") is True)
r = client.get("/feedback/export")
check("feedback export still works + has the rated exchange",
      r.status_code == 200
      and "What is a falcon?" in r.get_data(as_text=True))
r = client.get("/feedback")
check("feedback page renders", r.status_code == 200)

# A database created before memory existed (no session_id column)
# is migrated in place: the column appears, old rows stay intact.
legacy_path = Path(_tmp.name) / "legacy-feedback.db"
conn = sqlite3.connect(legacy_path)
conn.execute(
    """CREATE TABLE exchanges (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           question TEXT NOT NULL,
           answer TEXT NOT NULL,
           rating TEXT,
           reason TEXT,
           created_at TEXT NOT NULL,
           rated_at TEXT
       )""")
conn.execute(
    "INSERT INTO exchanges (question, answer, created_at) "
    "VALUES ('Old question', 'Old answer', '2026-11-15T00:00:00')")
conn.commit()
conn.close()
old_db_path = appmod.DB_PATH
appmod.DB_PATH = legacy_path
try:
    mig_client = appmod.app.test_client()
    r = mig_client.post("/ask", json={"question": "What is a saxophone?"})
    check("legacy database -> /ask still 200 after migration",
          r.status_code == 200, f"got {r.status_code}")
finally:
    appmod.DB_PATH = old_db_path
conn = sqlite3.connect(legacy_path)
cols = [row[1] for row in conn.execute("PRAGMA table_info(exchanges)")]
old_row = conn.execute(
    "SELECT question, answer FROM exchanges WHERE id = 1").fetchone()
conn.close()
check("migration added the session_id column", "session_id" in cols)
check("migration kept the old row intact",
      old_row == ("Old question", "Old answer"), f"got {old_row}")

print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
