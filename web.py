import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from duckduckgo_search import DDGS


_executor = ThreadPoolExecutor(max_workers=6)
LLM_TIMEOUT_SECONDS = 45.0
SEARCH_RESULTS_PER_QUERY = 5
FINAL_TOP_K = 5

TRUSTED_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "nytimes.com",
    "wsj.com",
    "bloomberg.com",
    "ft.com",
    "who.int",
    "un.org",
    "worldbank.org",
    "imf.org",
    "nasa.gov",
    "nih.gov",
    "europa.eu",
    "india.gov.in",
    "nseindia.com",
    "bseindia.com",
    "sec.gov",
}

BLOCKED_DOMAIN_FRAGMENTS = {
    "pinterest.",
    "quora.",
    "reddit.",
    "taboola.",
    "outbrain.",
}


def _domain_of(url):
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _is_blocked_domain(domain):
    return any(x in domain for x in BLOCKED_DOMAIN_FRAGMENTS)


async def _call_ollama_chat(messages, ollama_url, model_name, headers=None):
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
        res = await client.post(
            ollama_url,
            json={"model": model_name, "messages": messages, "stream": False},
            headers=headers or {},
        )
        res.raise_for_status()
        data = res.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


async def analyze_query_intent(system_prompt, user_query, ollama_url, model_name, headers=None):
    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""
Analyze this user query:

{user_query}

Return STRICT JSON only:
{{
  "needs_search": true/false,
  "needs_fresh_data": true/false,
  "confidence": 0-100
}}
""",
        },
    ]
    try:
        content = await _call_ollama_chat(prompt, ollama_url, model_name, headers=headers)
        return json.loads(content)
    except Exception:
        return {"needs_search": True, "needs_fresh_data": False, "confidence": 50}


async def rewrite_query_for_search(system_prompt, user_query, ollama_url, model_name, headers=None):
    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""
You are a professional search query optimizer.

User Input:
{user_query}

Rewrite this into the best possible web search query.

Rules:
- Make it specific
- Add relevant keywords
- Add year if time-sensitive
- Remove unnecessary words
- Return ONLY the final optimized query
""",
        },
    ]
    try:
        content = await _call_ollama_chat(prompt, ollama_url, model_name, headers=headers)
        return content or user_query
    except Exception:
        return user_query


async def expand_queries(system_prompt, user_query, ollama_url, model_name, headers=None):
    current_year = datetime.now(timezone.utc).year
    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""
Generate exactly 3 web search query variants for the input.

Input:
{user_query}

Return STRICT JSON:
{{
  "queries": [
    "specific version",
    "broader version",
    "freshness focused version with {current_year}"
  ]
}}
""",
        },
    ]
    try:
        content = await _call_ollama_chat(prompt, ollama_url, model_name, headers=headers)
        payload = json.loads(content)
        queries = [str(q).strip() for q in payload.get("queries", []) if str(q).strip()]
        if len(queries) >= 3:
            return queries[:3]
    except Exception:
        pass

    base = await rewrite_query_for_search(system_prompt, user_query, ollama_url, model_name, headers=headers)
    return [
        base,
        user_query.strip(),
        f"{user_query.strip()} latest {current_year}",
    ]


def _duckduckgo_search(query, max_results=SEARCH_RESULTS_PER_QUERY):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "engine": "DuckDuckGo",
                    "query": query,
                    "title": r.get("title"),
                    "link": r.get("href"),
                    "snippet": r.get("body"),
                }
            )
    return results


async def async_duckduckgo(query, max_results=SEARCH_RESULTS_PER_QUERY):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _duckduckgo_search, query, max_results)


async def parallel_search(queries):
    tasks = [async_duckduckgo(q, SEARCH_RESULTS_PER_QUERY) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    combined = []
    for r in results:
        if isinstance(r, list):
            combined.extend(r)
    return combined


def filter_and_dedupe_results(results):
    filtered = []
    seen_links = set()
    for r in results:
        link = (r.get("link") or "").strip()
        title = (r.get("title") or "").strip()
        if not link or not title or link in seen_links:
            continue
        domain = _domain_of(link)
        if _is_blocked_domain(domain):
            continue
        seen_links.add(link)
        filtered.append(r)
    return filtered


def _heuristic_score(query, result):
    score = 0.0
    q_tokens = {w.lower() for w in (query or "").split() if len(w) > 2}
    hay = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
    overlap = sum(1 for t in q_tokens if t in hay)
    score += min(overlap * 1.2, 6.0)

    domain = _domain_of(result.get("link", ""))
    if any(domain.endswith(td) for td in TRUSTED_DOMAINS):
        score += 2.5
    if result.get("snippet"):
        score += 0.5
    return round(score, 2)


async def rerank_results(system_prompt, user_query, results, ollama_url, model_name, headers=None):
    if not results:
        return []

    compact = []
    for i, r in enumerate(results, 1):
        compact.append(
            {
                "id": i,
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", ""),
            }
        )

    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""
Rank these search results for relevance and trustworthiness for the query.

Query:
{user_query}

Results JSON:
{json.dumps(compact, ensure_ascii=False)}

Return STRICT JSON only:
{{
  "scores": [
    {{"id": 1, "score": 1-10}},
    {{"id": 2, "score": 1-10}}
  ]
}}
""",
        },
    ]

    scored = []
    try:
        content = await _call_ollama_chat(prompt, ollama_url, model_name, headers=headers)
        parsed = json.loads(content)
        score_map = {}
        for s in parsed.get("scores", []):
            try:
                idx = int(s.get("id"))
                val = float(s.get("score"))
                score_map[idx] = max(1.0, min(10.0, val))
            except Exception:
                continue
        for i, r in enumerate(results, 1):
            llm_score = score_map.get(i, 0.0)
            heuristic = _heuristic_score(user_query, r)
            total = round(llm_score * 0.7 + heuristic * 0.3, 2)
            scored.append((total, r))
    except Exception:
        for r in results:
            scored.append((_heuristic_score(user_query, r), r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:FINAL_TOP_K]]


def build_search_messages(system_prompt, original_query, search_results):
    combined_text = ""
    for idx, r in enumerate(search_results, 1):
        combined_text += (
            f"\n[{idx}] Engine: {r.get('engine', 'Unknown')}\n"
            f"Title: {r.get('title', '')}\n"
            f"Snippet: {r.get('snippet', '')}\n"
            f"Link: {r.get('link', '')}\n"
        )

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""
Answer the user's question using the verified web results below.

User Question:
{original_query}

Web Results:
{combined_text}

Instructions:
- Give structured answer
- Use citation numbers like [1], [2]
- Do not invent facts
- Be concise but informative
""",
        },
    ]


async def intelligent_search_pipeline(
    system_prompt,
    user_query,
    ollama_url,
    model_name,
    headers=None,
):
    analysis = await analyze_query_intent(
        system_prompt,
        user_query,
        ollama_url,
        model_name,
        headers=headers,
    )
    if not analysis.get("needs_search", True):
        return None

    queries = await expand_queries(
        system_prompt,
        user_query,
        ollama_url,
        model_name,
        headers=headers,
    )

    raw_results = await parallel_search(queries)
    if not raw_results:
        return None

    filtered = filter_and_dedupe_results(raw_results)
    if not filtered:
        return None

    reranked = await rerank_results(
        system_prompt,
        user_query,
        filtered,
        ollama_url,
        model_name,
        headers=headers,
    )
    if not reranked:
        return None

    return build_search_messages(system_prompt, user_query, reranked)
