# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""A text-protocol Wikipedia search tool for the TT chat client.

The model asks for a search with a line ``SEARCH: <query>`` or reads an
article with ``OPEN: <title>``; the client (``search_rounds`` model arg) answers
with the results as the next user turn. A plain-text protocol rather than
OpenAI tool calls, so it works on servers without a tool-call parser. The
backend is the keyless MediaWiki API of English Wikipedia. No lmms-eval import,
so it is unit-testable.
"""

import json
import re
import urllib.parse
import urllib.request

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "tt-inference-server-evals/1.0 (https://github.com/tenstorrent/tt-inference-server)"
)
SEARCH_RESULTS = 5
SUMMARY_CHARS = 600
ARTICLE_CHARS = 6000
HTTP_TIMEOUT = 30

INSTRUCTIONS = """You have a Wikipedia search tool. To use it, reply with a single line and nothing else:
SEARCH: <query>
to get the top Wikipedia articles for the query with their summaries, or
OPEN: <exact article title>
to read the start of an article. The results come back in the next message. You may use the tool up to {rounds} times. When you are ready, reply with your final answer in the required format."""

LIMIT_REACHED = (
    "The search limit is reached. Give your final answer now, in the required format."
)

# One tool request per reply: the first SEARCH/OPEN line. A reply that already
# carries a final-answer line is final even if it also mentions the tool.
_REQUEST_RE = re.compile(
    r"^[\s*`]*(SEARCH|OPEN)[\s*`]*:[\s*`]*(.+?)[\s*`]*$", re.MULTILINE
)
_FINAL_RE = re.compile(
    r"^[\s*_#]*(Exact Answer|Answer)[\s*_]*:", re.MULTILINE | re.IGNORECASE
)


def find_request(text: str):
    """(``"SEARCH"`` | ``"OPEN"``, argument) for a tool request, else None."""
    if not text or _FINAL_RE.search(text):
        return None
    match = _REQUEST_RE.search(text)
    if not match or not match.group(2).strip():
        return None
    return match.group(1).upper(), match.group(2).strip()


def _api(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    request = urllib.request.Request(
        f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as reply:
        return json.load(reply)


def search(query: str) -> str:
    data = _api(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": SEARCH_RESULTS,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
        }
    )
    pages = sorted(
        (data.get("query") or {}).get("pages") or [], key=lambda p: p.get("index", 0)
    )
    if not pages:
        return f"No Wikipedia results for: {query}"
    lines = [f"Wikipedia results for: {query}"]
    for n, page in enumerate(pages, 1):
        summary = " ".join((page.get("extract") or "").split())[:SUMMARY_CHARS]
        lines.append(f"{n}. {page['title']}\n{summary}")
    return "\n\n".join(lines)


def open_article(title: str) -> str:
    data = _api(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        }
    )
    pages = (data.get("query") or {}).get("pages") or []
    if not pages or pages[0].get("missing") or not pages[0].get("extract"):
        return f"No Wikipedia article titled: {title}"
    text = pages[0]["extract"].strip()
    if len(text) > ARTICLE_CHARS:
        text = text[:ARTICLE_CHARS] + "\n[... truncated]"
    return f"Wikipedia article: {pages[0]['title']}\n\n{text}"


def run_request(kind: str, argument: str) -> str:
    """The tool result for one request; a failed lookup is reported to the
    model rather than raised, so one network error does not end the question."""
    try:
        return search(argument) if kind == "SEARCH" else open_article(argument)
    except Exception as e:  # network / API errors
        return f"The {kind.lower()} request failed: {e}"
