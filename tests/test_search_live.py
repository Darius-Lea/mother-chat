"""Live network test for the MOTHER search module.

Hits the real search backends. Run explicitly, from the repo root:

    python tests/test_search_live.py

This is NOT part of the offline suite (tests/test_search.py): it
needs internet, and search providers can block or rate-limit any
given IP. A failure here means "this network cannot reach a search
backend right now", not necessarily "the code is broken" — check
which backend failed before concluding anything.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import search  # noqa: E402

failures = 0


def live_check(name, query, must_contain):
    global failures
    results = search.search_web(query, max_results=3, timeout=6)
    print(f"\n{name}: {query!r} -> {len(results)} result(s)")
    for r in results:
        print(f"  - {r['title']}")
        print(f"    {r['snippet'][:160]}")
        print(f"    {r['url']}")
    blob = " ".join(
        (r["title"] + " " + r["snippet"]).lower() for r in results)
    if results and must_contain.lower() in blob:
        print("  ok: results mention", must_contain)
    else:
        failures += 1
        print("  FAIL: expected results mentioning", must_contain)


live_check("general knowledge", "what is the capital of france", "Paris")
live_check("security (CVE)", "CVE-2024-3094 xz backdoor", "CVE")

print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} failure(s)")
sys.exit(1 if failures else 0)
