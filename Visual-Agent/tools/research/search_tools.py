import time
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

@tool
def arxiv(query: str) -> str:
    """Search for research papers on arXiv. Input should be a plain-text search query."""
    try:
        import arxiv as _arxiv
    except ImportError:
        return "arXiv integration is not installed."
    arxiv_client = _arxiv.Client()
    search = _arxiv.Search(
        query=query,
        max_results=5,
        sort_by=_arxiv.SortCriterion.Relevance,
    )
    for attempt in range(3):
        try:
            results = list(arxiv_client.results(search))
            if not results:
                return "No papers found for that query."
            parts = []
            for r in results[:2]:
                authors = ", ".join(str(a) for a in r.authors[:3])
                parts.append(
                    f"Title: {r.title}\n"
                    f"Authors: {authors}\n"
                    f"Summary: {r.summary[:600]}\n"
                    f"URL: {r.entry_id}"
                )
            return "\n\n---\n\n".join(parts)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "too many requests" in msg.lower():
                if attempt < 2:
                    time.sleep(12 * (attempt + 1))
                    continue
                return "arXiv rate-limited after retries. Please try again later."
            return f"arXiv error: {msg}"
    return "arXiv search failed."


@tool
def wikipedia(query: str) -> str:
    """Search Wikipedia for optional background information, never exam statistics."""
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
    return runner.invoke(query)
