"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads.
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree  # arXiv answers with Atom XML

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Wait = the server's Retry-After when given, else exponential backoff base * 2**attempt plus jitter; always capped.
    The last failed attempt re-raises without sleeping. Any other exception propagates immediately.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt >= attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = exc.retry_after
            else:
                delay = min(base * 2 ** attempt, cap)
                delay += random.uniform(0, delay * 0.25)
            time.sleep(max(0.0, min(delay, cap)))


RETRY_STATUS = {429, 500, 502, 503, 504}
TIMEOUT = 30.0


def _retry_after(response):
    try:
        return float(response.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def _http(method, url, **kwargs):
    """One HTTP call: RetryableError for 429/5xx and network errors, httpx.HTTPStatusError for other 4xx."""
    try:
        response = httpx.request(method, url, timeout=TIMEOUT, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRY_STATUS:
        raise RetryableError(f"HTTP {response.status_code}", retry_after=_retry_after(response))
    response.raise_for_status()
    return response


def _clean(text, limit=None):
    text = " ".join(str(text or "").split())
    return text[:limit] if limit else text


def _clamp(value, low, high):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = low
    return max(low, min(high, value))


def _error(exc):
    """'ERROR: ...' string with the Exa key (which travels in the URL) redacted."""
    text = f"ERROR: {type(exc).__name__}: {exc}"
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key:
        text = text.replace(key, "***")
    return re.sub(r"(exaApiKey=)[^&\s'\"]+", r"\1***", text)[:500]


# ---- arXiv ----
_ARXIV_LAST = 0.0
_ARXIV_LOCK = threading.Lock()
_ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _arxiv_get(params):
    global _ARXIV_LAST
    with _ARXIV_LOCK:  # parallel researchers share one arXiv client: >= 3 s between requests
        wait = 3.0 - (time.monotonic() - _ARXIV_LAST)
        if wait > 0:
            time.sleep(wait)
        try:
            return _http("GET", ARXIV_URL, params=params)
        finally:
            _ARXIV_LAST = time.monotonic()


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by a few plain keywords (no quotes or operators), newest first.
    Returns a JSON list of {id, url, published, title, summary}. Good for primary research papers and recent preprints."""
    try:
        terms = [t for t in re.findall(r"[A-Za-z0-9\-]+", query or "")
                 if t.strip("-") and t.upper() not in {"AND", "OR", "NOT"}]
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
                  "sortOrder": "descending", "max_results": _clamp(max_results, 1, 30), "start": 0}
        response = with_retry(lambda: _arxiv_get(params), attempts=6, base=3.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(response.text)
        records = []
        for entry in root.findall("a:entry", _ATOM):
            raw_id = _clean(entry.findtext("a:id", "", _ATOM))
            if "/abs/" not in raw_id:
                continue
            paper_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[1])
            records.append({
                "id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                "published": _clean(entry.findtext("a:published", "", _ATOM))[:10],
                "title": _clean(entry.findtext("a:title", "", _ATOM)),
                "summary": _clean(entry.findtext("a:summary", "", _ATOM), 600)})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001 - tools never raise
        return _error(exc)


# ---- Hugging Face ----
def _hf_records(items, prefer_ai_summary=False):
    records = []
    for item in items if isinstance(items, list) else []:
        item = item or {}
        paper = item.get("paper") or {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        summary = (paper.get("ai_summary") if prefer_ai_summary else None) or paper.get("summary") or item.get("summary")
        records.append({
            "id": paper_id, "url": f"https://huggingface.co/papers/{paper_id}",
            "published": _clean(paper.get("publishedAt") or item.get("publishedAt"))[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _clean(summary, 600),
            "upvotes": paper.get("upvotes") or 0,
            "github": paper.get("githubRepo"), "stars": paper.get("githubStars")})
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": _clamp(limit, 1, 100)}
        if date:
            params["date"] = date
        response = with_retry(lambda: _http("GET", HF_DAILY_URL, params=params))
        records = _hf_records(response.json())
        if keyword:
            needle = keyword.lower()
            records = [r for r in records if needle in (r["title"] + " " + r["summary"]).lower()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        if not (query or "").strip():
            return "NO RESULTS"
        params = {"q": query.strip(), "limit": _clamp(limit, 1, 50)}
        response = with_retry(lambda: _http("GET", HF_SEARCH_URL, params=params))
        records = _hf_records(response.json(), prefer_ai_summary=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- web search / fetch through the Exa MCP endpoint ----
def _is_rate_limited(text, meta):
    """Exa's free tier may answer HTTP 200 with a rate-limit message and/or a flag in result._meta."""
    if isinstance(meta, dict):
        for key, value in meta.items():
            name = str(key).lower()
            if "rate" in name and value:
                return True
    return "rate limit" in (text or "").lower()[:600]


def _exa_call(tool_name, arguments):
    url = EXA_URL
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key:
        url = f"{EXA_URL}?exaApiKey={key}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def once():
        response = _http("POST", url, json=payload, headers=headers)
        body = None
        for line in response.text.splitlines():
            if line.startswith("data:"):
                try:
                    body = json.loads(line[5:].strip())
                    break
                except ValueError:
                    continue
        if body is None:
            try:
                body = response.json()
            except ValueError as exc:
                raise RuntimeError("unreadable Exa response") from exc
        if "error" in body:
            message = str((body["error"] or {}).get("message", body["error"]))
            if _is_rate_limited(message, None):
                raise RetryableError("Exa rate limit", retry_after=20.0)
            raise RuntimeError(f"Exa error: {message[:300]}")
        result = body.get("result") or {}
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        if _is_rate_limited(text, result.get("_meta")):
            raise RetryableError("Exa rate limit", retry_after=20.0)
        if result.get("isError"):
            raise RuntimeError(f"Exa tool error: {text[:300]}")
        return text

    return with_retry(once, attempts=6, base=5.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        if not (query or "").strip():
            return "NO RESULTS"
        args = {"query": query.strip(), "objective": (objective or "").strip() or f"find pages about: {query.strip()}",
                "numResults": _clamp(num_results, 1, 10)}
        text = _exa_call("web_search_exa", args).strip()
        return text[:12000] if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        if not (url or "").strip():
            return "NO RESULTS"
        text = _exa_call("web_fetch_exa", {"urls": [url.strip()]}).strip()
        return text[:12000] if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
