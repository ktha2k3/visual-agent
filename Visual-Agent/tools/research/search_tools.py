import json
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from utils.config import get_settings


@tool
def current_datetime() -> str:
    """Return the application's current local date, time, timezone, and UTC offset."""
    timezone_name = get_settings().app_timezone
    current = datetime.now(ZoneInfo(timezone_name))
    return current.isoformat() + f" ({timezone_name})"


def _normalize_query(query: str) -> str:
    return " ".join(query.split())[:500]


@tool
def tavily(query: str) -> str:
    """Search the web with Tavily for trustworthy STEM concepts and worked examples."""
    normalized = _normalize_query(query)
    if not normalized:
        return "[]"
    try:
        from langchain_tavily import TavilySearch

        response = TavilySearch(
            max_results=5,
            search_depth="advanced",
            topic="general",
            include_answer=False,
            include_raw_content=False,
            include_images=False,
        ).invoke({"query": normalized})
    except Exception as exc:
        return json.dumps(
            {"error": f"Tavily search unavailable: {type(exc).__name__}"},
            ensure_ascii=False,
        )

    results = response.get("results", []) if isinstance(response, dict) else []
    payload = [
        {
            "title": str(item.get("title", ""))[:300],
            "snippet": str(item.get("content", ""))[:1200],
            "url": str(item.get("url", ""))[:2000],
            "score": item.get("score"),
        }
        for item in results
        if isinstance(item, dict) and item.get("url")
    ]
    return json.dumps(payload, ensure_ascii=False)


@tool
def arxiv(query: str) -> str:
    """Search arXiv for academic STEM papers relevant to a problem or method."""
    normalized = _normalize_query(query)
    if not normalized:
        return "No arXiv query was provided."
    try:
        from langchain_community.tools import ArxivQueryRun
        from langchain_community.utilities import ArxivAPIWrapper

        runner = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(
                top_k_results=3,
                load_max_docs=3,
                doc_content_chars_max=3000,
            )
        )
        return runner.invoke(normalized)
    except Exception as exc:
        return f"arXiv search unavailable: {type(exc).__name__}"


@tool
def wikipedia(query: str) -> str:
    """Search Wikipedia for optional STEM background information."""
    normalized = _normalize_query(query)
    if not normalized:
        return "No Wikipedia query was provided."
    try:
        import wikipedia as wiki_package
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
    except ImportError:
        return "Wikipedia integration is not installed."
    wiki_package.set_user_agent(
        "LiAn-Visual-Agent/0.1 (contact: trankimquang1603@gmail.com)"
    )
    runner = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
    )
    try:
        return runner.invoke(normalized)
    except Exception as exc:
        return f"Wikipedia search unavailable: {type(exc).__name__}"
