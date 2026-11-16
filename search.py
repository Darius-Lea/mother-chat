"""Internet search for MOTHER, no API key needed.

When a question needs current or external facts (news, prices, what
happened today, a CVE, anything she could not have learned in
training), the app searches the web and feeds the top results into
her prompt as context, so she answers grounded in them.

How it works
------------
search_web() tries three no-key HTML search backends in order and
returns the first non-empty result set:

  1. DuckDuckGo HTML   (html.duckduckgo.com)
  2. Bing HTML         (www.bing.com/search)
  3. DuckDuckGo Lite   (lite.duckduckgo.com)

Every failure is silent: a timeout, a block page, a parse miss, or no
internet at all just returns an empty list, and the caller answers
without search context. The user never sees a search error.

Prompt integration
------------------
MOTHER's verified prompt is ``Lesson: {question}\\nExplanation:``.
The search context is folded into the question string itself by
build_grounded_question(), so the prompt becomes::

    Lesson: {question}
    Context: {search results}
    Explanation:

No change to mother_model.py is needed, and the question stored in
the feedback database stays the original question (only the string
handed to generate() carries the context). The context is capped at
600 characters so the whole prompt stays well under her 512-token
context window (question <= 500 chars + context <= 600 chars is
roughly 275 tokens at most).

One honest limit: MOTHER was never trained with a Context line in
her prompt, so grounding is best-effort until a future training
round teaches her to use context. The context is there in her
window and she conditions on it; how faithfully she follows it is
measured, not assumed. See README.md.
"""

import base64
import html as html_lib
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Words and phrases that mean "this needs current or external facts".
# Word-boundary matching, so "news" does not fire on "newspaper".
_SEARCH_TRIGGERS = [
    "today", "tonight", "yesterday", "this week", "this month",
    "this year", "right now", "current", "currently", "latest",
    "recent", "recently", "news", "just happened", "what happened",
    "who won", "price", "stock", "weather", "forecast", "score",
    "search", "look up", "look it up", "google it",
    "on the internet", "on the web", "online",
    "2024", "2025", "2026",
]
_TRIGGER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _SEARCH_TRIGGERS) + r")\b",
    re.IGNORECASE,
)
_CVE_RE = re.compile(r"\bcve-\d{4}-\d+\b", re.IGNORECASE)


def needs_search(question):
    """True when the question likely needs current or external facts."""
    if not question:
        return False
    return bool(_TRIGGER_RE.search(question) or _CVE_RE.search(question))


# ------------------------------------------------------------------ net --
def _fetch(url, timeout):
    """GET a page, return its decoded text, or None on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            raw = resp.read()
    except (urllib.error.URLError, socket.timeout, OSError, ValueError):
        return None
    except Exception:  # noqa: BLE001 - search must never raise
        return None
    charset = "utf-8"
    try:
        content_type = resp.headers.get_content_charset()
        if content_type:
            charset = content_type
    except Exception:  # noqa: BLE001 - fall back to utf-8
        pass
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


def _clean(text):
    """Strip tags, unescape entities, collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    # Tags stripped above leave "word ," gaps; close them up.
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    return text


def _dedupe(results):
    seen = set()
    out = []
    for r in results:
        key = r["title"].lower().strip()
        if key and key not in seen and r["snippet"]:
            seen.add(key)
            out.append(r)
    return out


# -------------------------------------------------------------- parsers --
def _ddg_target_url(href):
    """DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<target>."""
    if not href:
        return ""
    try:
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        target = qs.get("uddg", [""])[0]
        if target:
            return target
        if href.startswith("http"):
            return href
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return ""


def parse_ddg_html(page):
    """Parse html.duckduckgo.com results: result__a + result__snippet."""
    titles = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        page, re.S | re.I)
    snippets = re.findall(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        page, re.S | re.I)
    results = []
    for (href, title), snippet in zip(titles, snippets):
        results.append({
            "title": _clean(title),
            "snippet": _clean(snippet),
            "url": _ddg_target_url(html_lib.unescape(href)),
        })
    return _dedupe(results)


def parse_ddg_lite(page):
    """Parse lite.duckduckgo.com results: result-link + result-snippet."""
    titles = re.findall(
        r"""<a[^>]*class=['"]result-link['"][^>]*href=['"]([^'"]*)['"][^>]*>(.*?)</a>""",
        page, re.S | re.I)
    snippets = re.findall(
        r"""<td[^>]*class=['"]result-snippet['"][^>]*>(.*?)</td>""",
        page, re.S | re.I)
    results = []
    for (href, title), snippet in zip(titles, snippets):
        results.append({
            "title": _clean(title),
            "snippet": _clean(snippet),
            "url": html_lib.unescape(href),
        })
    return _dedupe(results)


def _bing_target_url(href):
    """Bing wraps links as bing.com/ck/a?...&u=<base64 target>."""
    if not href:
        return ""
    try:
        parsed = urllib.parse.urlparse(html_lib.unescape(href))
        qs = urllib.parse.parse_qs(parsed.query)
        encoded = qs.get("u", [""])[0]
        if encoded:
            padded = encoded + "=" * (-len(encoded) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode(
                "utf-8", errors="ignore")
            # Some Bing links carry a u= value that is not a URL at
            # all; a decode that is not an http(s) URL is garbage.
            if decoded.startswith(("http://", "https://")):
                return decoded
            return ""
        if href.startswith("http"):
            return href
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return ""


def parse_bing(page):
    """Parse www.bing.com/search results: b_algo blocks."""
    results = []
    for block in re.findall(
            r'<li class="b_algo".{0,6000}?</li>', page, re.S | re.I):
        m_title = re.search(
            r"<h2><a[^>]*href=\"([^\"]*)\"[^>]*>(.*?)</a></h2>",
            block, re.S | re.I)
        m_snippet = re.search(
            r'<div class="b_caption"><p[^>]*>(.*?)</p>',
            block, re.S | re.I)
        if not m_title or not m_snippet:
            continue
        url = _bing_target_url(m_title.group(1))
        if not url:
            m_cite = re.search(r"<cite>(.*?)</cite>", block, re.S | re.I)
            if m_cite:
                url = _clean(m_cite.group(1)).split(" ")[0]
        results.append({
            "title": _clean(m_title.group(2)),
            "snippet": _clean(m_snippet.group(1)),
            "url": url,
        })
    return _dedupe(results)


# -------------------------------------------------------------- search ---
_BACKENDS = [
    ("ddg_html",
     "https://html.duckduckgo.com/html/?q={q}", parse_ddg_html),
    ("bing",
     "https://www.bing.com/search?q={q}&count=5", parse_bing),
    ("ddg_lite",
     "https://lite.duckduckgo.com/lite/?q={q}", parse_ddg_lite),
]


def search_web(query, max_results=3, timeout=4, backends=None):
    """Search the web. Returns a list of {title, snippet, url}.

    Tries each backend in order and returns the first non-empty
    result set. Returns [] on any failure (timeout, block page,
    no internet). Never raises.
    """
    query = (query or "").strip()
    if not query:
        return []
    encoded = urllib.parse.quote_plus(query[:300])
    for _name, url_template, parser in (backends or _BACKENDS):
        try:
            page = _fetch(url_template.format(q=encoded), timeout)
            if not page:
                continue
            results = parser(page)
        except Exception:  # noqa: BLE001 - one backend must not kill search
            continue
        if results:
            return results[:max_results]
    return []


# ------------------------------------------------------- prompt context --
def build_context(results, max_chars=600):
    """Format search results as a short context block for the prompt."""
    lines = []
    total = 0
    for r in results or []:
        line = f"{r['title']}: {r['snippet']}"
        line = line[:220]
        if total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return " | ".join(lines)


def build_grounded_question(question, context):
    """Fold search context into the question string for generate().

    generate() builds ``Lesson: {q}\\nExplanation:``, so passing
    this string as the question yields::

        Lesson: {question}
        Context: {context}
        Explanation:

    The original question is untouched (that is what gets stored in
    the feedback database); only the string handed to generate()
    carries the context.
    """
    if not context:
        return question
    return f"{question}\nContext: {context}"
