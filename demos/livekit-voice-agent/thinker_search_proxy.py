"""Web-search proxy in front of the QwQ Thinker.

The MiniCPM-o gateway calls THINKER_BASE_URL/v1/chat/completions. Point it at this
proxy (:8000) with the real QwQ moved to :8001. For each chat request the proxy:

  1. extracts the user's actual question from the (gateway-wrapped) prompt,
  2. if it needs current/external facts (needs_search), runs a DuckDuckGo search,
  3. injects the results as a system message,
  4. forwards everything to the real QwQ and returns its response.

This gives the Thinker live web facts (e.g. today's NVDA price) without changing the
gateway or QwQ. Run in the thinker-venv (has ddgs/httpx/fastapi/uvicorn).
"""
from __future__ import annotations

import asyncio
import logging
import re

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

QWQ_URL = "http://localhost:8001"
SEARCH_MAX = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [proxy] %(message)s")
log = logging.getLogger("thinker-proxy")

_NEEDS_SEARCH = re.compile(
    r"\b(latest|current(ly)?|today|tonight|right now|as of|recent(ly)?|now\b|"
    r"this (year|week|month)|news|update|price|stock|shares?|worth|cost|quote|"
    r"who (is|are|won|leads)|when (did|does|will|is)|where is|score|weather|"
    r"forecast|standings|election|release[d]?|version|20(2[4-9]|3[0-9]))\b",
    re.IGNORECASE,
)


def needs_search(text: str) -> bool:
    return bool(_NEEDS_SEARCH.search(text or ""))


# --- Live finance quotes (stock prices snippet-search can't reliably give) ---
_STOCK_CTX = re.compile(r"\b(stock|shares?|price|quote|trading|ticker|market cap|worth)\b", re.IGNORECASE)
_NAME_TO_TICKER = {
    "nvidia": "NVDA", "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "intel": "INTC", "netflix": "NFLX", "amd": "AMD",
    "palantir": "PLTR", "broadcom": "AVGO", "bitcoin": "BTC-USD", "ethereum": "ETH-USD",
}
_NOT_TICKERS = {"USD", "CEO", "API", "AI", "USA", "ETF", "IPO", "NYSE", "NASDAQ"}


def find_tickers(text: str) -> list[str]:
    found: list[str] = []
    low = text.lower()
    for name, tk in _NAME_TO_TICKER.items():
        if re.search(r"\b" + re.escape(name) + r"\b", low) and tk not in found:
            found.append(tk)
    for m in re.findall(r"\b[A-Z]{2,5}\b", text):
        if m not in found and m not in _NOT_TICKERS:
            found.append(m)
    return found[:3]


async def fetch_quote(ticker: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": "Mozilla/5.0"}) as c:
            r = await c.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"interval": "1d", "range": "1d"},
            )
            d = r.json()
        m = d["chart"]["result"][0]["meta"]
        price = m.get("regularMarketPrice")
        if price is None:
            return None
        cur = m.get("currency", "USD")
        prev = m.get("chartPreviousClose")
        return f"{ticker} = {price} {cur} (previous close {prev})"
    except Exception as e:  # noqa: BLE001
        log.warning("quote %s failed: %s", ticker, e)
        return None


def extract_question(content: str) -> str:
    """Pull the real question out of the gateway/frontend prompt wrappers."""
    if not isinstance(content, str):
        return ""
    m = re.search(r"voice demo:\s*\n(.+?)\n\nReturn", content, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"Current user request:\s*\n(.+?)(\n|$)", content, re.S)
    if m:
        return m.group(1).strip()
    return content.strip()


def ddgs_search(query: str, n: int) -> list[dict]:
    from ddgs import DDGS
    out: list[dict] = []
    with DDGS() as d:
        for r in d.text(query, max_results=n):
            out.append({
                "title": r.get("title", ""),
                "url": r.get("href", "") or r.get("url", ""),
                "snippet": r.get("body", "") or r.get("snippet", ""),
            })
    return out


app = FastAPI(title="Thinker Search Proxy")


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{QWQ_URL}/v1/models")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])

    question = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            question = extract_question(m.get("content", ""))
            break

    insert_at = max(0, len(messages) - 1)

    # Authoritative live quotes for stock/crypto questions (Yahoo Finance).
    if question and _STOCK_CTX.search(question):
        quotes = [q for q in [await fetch_quote(t) for t in find_tickers(question)] if q]
        if quotes:
            messages.insert(insert_at, {
                "role": "system",
                "content": "Live market quotes (authoritative, state these as the current price):\n"
                           + "\n".join(quotes),
            })
            body["messages"] = messages
            log.info("FINANCE q=%r -> %s", question[:60], quotes)

    if question and needs_search(question):
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, ddgs_search, question[:200], SEARCH_MAX
            )
        except Exception as e:  # noqa: BLE001
            log.warning("search failed: %s", e)
            results = []
        if results:
            ctx = "Up-to-date web search results — use these for any current facts:\n" + "\n".join(
                f"[{i+1}] {r['title']}: {r['snippet'][:300]} ({r['url']})"
                for i, r in enumerate(results)
            )
            insert_at = max(0, len(messages) - 1)
            messages.insert(insert_at, {"role": "system", "content": ctx})
            body["messages"] = messages
            log.info("SEARCH q=%r -> %d results, injected", question[:80], len(results))
        else:
            log.info("SEARCH q=%r -> 0 results", question[:80])
    else:
        log.info("no search for q=%r", (question or "")[:80])

    async with httpx.AsyncClient(timeout=180.0) as c:
        r = await c.post(f"{QWQ_URL}/v1/chat/completions", json=body)
    return JSONResponse(r.json(), status_code=r.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
