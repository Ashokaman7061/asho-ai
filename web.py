import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
from duckduckgo_search import DDGS


_executor = ThreadPoolExecutor(max_workers=6)


async def analyze_query_intent(system_prompt, user_query, ollama_url, model_name, headers=None):
    """
    Ask LLM whether search is required and detect intent.
    """
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

    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(
            ollama_url,
            json={
                "model": model_name,
                "messages": prompt,
                "stream": False,
            },
            headers=headers or {},
        )
        res.raise_for_status()
        data = res.json()

    content = data.get("message", {}).get("content", "").strip()
    try:
        return json.loads(content)
    except Exception:
        return {"needs_search": True, "needs_fresh_data": False, "confidence": 50}


async def rewrite_query_for_search(system_prompt, user_query, ollama_url, model_name, headers=None):
    """
    Convert vague or indirect user query into optimized web search query.
    """
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

    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post(
            ollama_url,
            json={
                "model": model_name,
                "messages": prompt,
                "stream": False,
            },
            headers=headers or {},
        )
        res.raise_for_status()
        data = res.json()

    return data.get("message", {}).get("content", "").strip()


def _duckduckgo_search(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "engine": "DuckDuckGo",
                    "title": r.get("title"),
                    "link": r.get("href"),
                    "snippet": r.get("body"),
                }
            )
    return results


async def async_duckduckgo(query, max_results=5):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _duckduckgo_search, query, max_results)


async def parallel_search(query):
    """
    Add more engines here later (Bing, Brave, APIs).
    """
    tasks = [async_duckduckgo(query, 5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    combined = []
    for r in results:
        if isinstance(r, list):
            combined.extend(r)
    return combined


def build_search_messages(system_prompt, original_query, search_results):
    combined_text = ""
    for idx, r in enumerate(search_results, 1):
        combined_text += (
            f"\n[{idx}] Engine: {r['engine']}\n"
            f"Title: {r['title']}\n"
            f"Snippet: {r['snippet']}\n"
            f"Link: {r['link']}\n"
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
    """
    Full intelligent search flow:
    1. Analyze query intent
    2. Decide search necessity
    3. Rewrite query
    4. Parallel search
    5. Build structured LLM input
    """
    analysis = await analyze_query_intent(
        system_prompt,
        user_query,
        ollama_url,
        model_name,
        headers=headers,
    )

    if not analysis.get("needs_search", True):
        return None

    optimized_query = await rewrite_query_for_search(
        system_prompt,
        user_query,
        ollama_url,
        model_name,
        headers=headers,
    )

    search_results = await parallel_search(optimized_query)
    if not search_results:
        return None

    return build_search_messages(system_prompt, user_query, search_results)
