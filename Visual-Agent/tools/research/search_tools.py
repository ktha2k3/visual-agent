import time

import arxiv as _arxiv
import wikipedia as _wiki_pkg
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool


# ── arXiv ─────────────────────────────────────────────────────────────────────
_arxiv_client = _arxiv.Client()


@tool
def arxiv(query: str) -> str:
    """Search for research papers on arXiv. Input should be a plain-text search query."""
    search = _arxiv.Search(
        query=query,
        max_results=5,
        sort_by=_arxiv.SortCriterion.Relevance,
    )
    for attempt in range(3):
        try:
            results = list(_arxiv_client.results(search))
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


# ── Wikipedia ─────────────────────────────────────────────────────────────────
_wiki_pkg.set_user_agent("LiAn-Visual-Agent/0.1 (contact: trankimquang1603@gmail.com)")

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000),
    description="Search Wikipedia for factual information. Input should be a search query.",
)
