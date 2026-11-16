"""Tests for the MOTHER internet-search feature.

Offline and deterministic: no network calls. Run from the repo root
with a Python that has flask + sentencepiece installed:

    python tests/test_search.py

Covers:
  * needs_search() trigger logic
  * the three HTML parsers (Bing on a real saved page, DDG on
    hand-written fixtures matching their known markup)
  * context building + truncation, grounded-question format
  * the search_web() fallback chain (with a stubbed fetcher)
  * the full Flask /ask flow with a stubbed model: search used when
    triggered, skipped otherwise, silent fallback when search raises,
    feedback + export + /health still working

The live network test is separate: tests/test_search_live.py.
"""

import os
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
os.environ["FEEDBACK_DB"] = str(Path(_tmp.name) / "test-feedback.db")

import search  # noqa: E402
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


# ------------------------------------------------------- needs_search ----
print("needs_search")
for q in [
    "What is the news today?",
    "Who won the game last night?",
    "What is the weather forecast?",
    "What happened in the world this week?",
    "What is the latest CVE-2024-3094 about?",
    "Search the web for the price of gold",
    "What is happening in 2026?",
    "Look up the stock price of Apple",
]:
    check(f"triggers: {q!r}", search.needs_search(q))
for q in [
    "What is a saxophone?",
    "Is a platypus a mammal?",
    "What number comes after 769?",
    "Hello. How are you?",
    "What is 7 + 5?",
    "My name is",
    "",
]:
    check(f"skips: {q!r}", not search.needs_search(q))
# Word boundaries: "news" inside "newspaper" must not trigger.
check("no trigger on 'newspaper'",
      not search.needs_search("What is a newspaper?"))

# ------------------------------------------------------------ parsers ----
print("parsers")
bing_page = (ROOT / "tests" / "fixtures" / "bing_sample.html").read_text()
bing_results = search.parse_bing(bing_page)
check("bing: >=2 results", len(bing_results) >= 2, f"got {len(bing_results)}")
check("bing: first title mentions Paris",
      bing_results and "Paris" in bing_results[0]["title"],
      f"got {bing_results[0]['title'] if bing_results else 'none'}")
check("bing: first snippet mentions France",
      bing_results and "France" in bing_results[0]["snippet"])
check("bing: first url decoded to a real page",
      bing_results and bing_results[0]["url"].startswith("http"),
      f"got {bing_results[0]['url'] if bing_results else 'none'}")
check("bing: no empty snippets",
      all(r["snippet"] for r in bing_results))
check("bing: empty page -> []", search.parse_bing("<html></html>") == [])
# Regression: a Bing u= value that decodes to non-URL junk must fall
# back to the cite URL instead of returning garbage (seen live).
junk_bing = (
    '<li class="b_algo"><h2><a target="_blank" '
    'href="https://www.bing.com/ck/a?!&amp;&amp;u=a1ZHbA&amp;ntb=1">'
    'NVD - Vulnerabilities</a></h2>'
    '<div class="b_caption"><p>Common Vulnerabilities and Exposures</p></div>'
    '<div class="b_attribution"><cite>https://nvd.nist.gov</cite></div></li>'
)
junk_res = search.parse_bing(junk_bing)
check("bing: garbage u= falls back to cite url",
      len(junk_res) == 1 and junk_res[0]["url"] == "https://nvd.nist.gov",
      f"got {junk_res[0]['url'] if junk_res else 'none'}")

ddg_html = """
<div class="result results_links">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FParis">Paris - Wikipedia</a>
  <a class="result__snippet"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FParis">Paris is the capital of France.</a>
</div>
<div class="result results_links">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fparis">Visit Paris</a>
  <a class="result__snippet"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fparis">A guide to Paris, city of light.</a>
</div>
"""
ddg_results = search.parse_ddg_html(ddg_html)
check("ddg html: 2 results", len(ddg_results) == 2,
      f"got {len(ddg_results)}")
check("ddg html: url decoded from uddg",
      ddg_results and ddg_results[0]["url"] == "https://en.wikipedia.org/wiki/Paris",
      f"got {ddg_results[0]['url'] if ddg_results else 'none'}")
check("ddg html: snippet kept",
      ddg_results and "capital of France" in ddg_results[0]["snippet"])
check("ddg html: empty page -> []",
      search.parse_ddg_html("<html></html>") == [])

ddg_lite = """
<tr><td><a class='result-link' href='https://en.wikipedia.org/wiki/Paris'>Paris - Wikipedia</a></td></tr>
<tr><td class='result-snippet'>Paris is the capital of France.</td></tr>
<tr><td><a class='result-link' href='https://example.com/paris'>Visit Paris</a></td></tr>
<tr><td class='result-snippet'>A guide to Paris, city of light.</td></tr>
"""
lite_results = search.parse_ddg_lite(ddg_lite)
check("ddg lite: 2 results", len(lite_results) == 2,
      f"got {len(lite_results)}")
check("ddg lite: url kept",
      lite_results and lite_results[0]["url"] == "https://en.wikipedia.org/wiki/Paris")
check("ddg lite: empty page -> []",
      search.parse_ddg_lite("<html></html>") == [])

# --------------------------------------------------- context + prompt ----
print("context + prompt")
sample = [
    {"title": "Paris - Wikipedia",
     "snippet": "Paris is the capital of France.",
     "url": "https://en.wikipedia.org/wiki/Paris"},
    {"title": "Visit Paris",
     "snippet": "A guide to Paris, city of light.",
     "url": "https://example.com/paris"},
]
ctx = search.build_context(sample)
check("build_context joins title + snippet",
      "Paris - Wikipedia: Paris is the capital of France." in ctx)
check("build_context empty -> ''", search.build_context([]) == "")
long_results = [{"title": "T" * 100, "snippet": "S" * 500, "url": ""}] * 5
check("build_context caps at 600 chars",
      len(search.build_context(long_results)) <= 600)
gq = search.build_grounded_question("What is the capital of France?", ctx)
check("grounded question format",
      gq == f"What is the capital of France?\nContext: {ctx}")
check("grounded question without context is unchanged",
      search.build_grounded_question("Hi", "") == "Hi")
# The prompt generate() will see (Lesson: {q}\nExplanation:).
prompt = f"Lesson: {gq}\nExplanation:"
check("prompt keeps the verified shape",
      prompt.startswith("Lesson: What is the capital of France?\n")
      and prompt.endswith("Explanation:")
      and "\nContext: " in prompt)

# --------------------------------------------------- fallback chain ------
print("fallback chain")
real_fetch = search._fetch
calls = []


def stub_fetch(url, timeout):
    calls.append(url)
    if "bing.com" in url:
        return bing_page
    return None  # DDG "blocked", lite "down"


search._fetch = stub_fetch
try:
    res = search.search_web("capital of france", max_results=3)
    check("chain skips a dead backend and uses Bing",
          len(res) >= 1 and "Paris" in res[0]["title"])
    check("chain tried DDG first",
          calls and "duckduckgo" in calls[0])
finally:
    search._fetch = real_fetch


def raising_fetch(url, timeout):
    raise RuntimeError("boom")


search._fetch = raising_fetch
try:
    check("search_web never raises", search.search_web("x") == [])
finally:
    search._fetch = real_fetch
check("search_web('') -> []", search.search_web("") == [])

# -------------------------------------------------------- Flask /ask -----
print("Flask /ask flow (stubbed model)")
client = appmod.app.test_client()

r = client.get("/health")
check("/health 200 + search_enabled",
      r.status_code == 200 and r.get_json().get("search_enabled") is True)

r = client.get("/")
check("index renders", r.status_code == 200 and b"MOTHER" in r.data)

real_search = appmod.search_web


def canned_search(question, max_results=3, timeout=4):
    return sample


appmod.search_web = canned_search
GENERATE_CALLS.clear()
r = client.post("/ask", json={"question": "What is the news today?"})
data = r.get_json()
check("trigger question -> 200 + searched=true",
      r.status_code == 200 and data.get("searched") is True, f"got {data}")
check("generate got the grounded question",
      GENERATE_CALLS
      and GENERATE_CALLS[0].startswith("What is the news today?\nContext: ")
      and "capital of France" in GENERATE_CALLS[0],
      f"got {GENERATE_CALLS[0][:120] if GENERATE_CALLS else 'none'}")
check("stored question is the original (no context leaked)",
      data.get("question") == "What is the news today?")
ex_id = data.get("exchange_id")

GENERATE_CALLS.clear()
r = client.post("/ask", json={"question": "What is a saxophone?"})
data2 = r.get_json()
check("plain question -> searched=false",
      r.status_code == 200 and data2.get("searched") is False)
check("generate got the plain question",
      GENERATE_CALLS and GENERATE_CALLS[0] == "What is a saxophone?")


def broken_search(question, max_results=3, timeout=4):
    raise RuntimeError("network down")


appmod.search_web = broken_search
GENERATE_CALLS.clear()
r = client.post("/ask", json={"question": "What is the news today?"})
data3 = r.get_json()
check("search raising -> still 200, searched=false",
      r.status_code == 200 and data3.get("searched") is False
      and data3.get("answer") == "STUB ANSWER")
check("generate still got the plain question",
      GENERATE_CALLS and GENERATE_CALLS[0] == "What is the news today?")
appmod.search_web = real_search

# Feedback + export still work on a searched exchange.
r = client.post("/feedback", json={
    "exchange_id": ex_id, "rating": "up", "reason": ""})
check("feedback POST still works",
      r.status_code == 200 and r.get_json().get("ok") is True)
r = client.get("/feedback/export")
check("feedback export still works + has the exchange",
      r.status_code == 200
      and "What is the news today?" in r.get_data(as_text=True))
r = client.get("/feedback")
check("feedback page renders", r.status_code == 200)
r = client.post("/ask", json={"question": ""})
check("empty question still 400", r.status_code == 400)

print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
