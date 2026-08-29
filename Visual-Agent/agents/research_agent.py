try:
    from langchain.agents import create_agent
except ImportError:  # LangChain 0.3 compatibility while migrating the lockfile
    from langgraph.prebuilt import create_react_agent as create_agent

from tools.research.search_tools import current_datetime
from utils.model_factory import get_chat_model

_PROMPT = (
    "You analyze structured, teacher-approved Vietnamese graduation exam documents "
    "provided by the learning supervisor. Always call current_datetime before deciding "
    "which exams are the two latest completed years. Never infer exam distribution from "
    "Wikipedia, arXiv, whole-subject score distributions, pending OCR, or an unfinished "
    "exam year. Preserve official source URLs, state missing coverage, and return Vietnamese."
)

try:
    research_agent = create_agent(
        model=get_chat_model("research"),
        tools=[current_datetime],
        system_prompt=_PROMPT,
        name="research_agent",
    )
except TypeError:  # create_react_agent on the legacy dependency set
    research_agent = create_agent(
        model=get_chat_model("research"),
        tools=[current_datetime],
        prompt=_PROMPT,
        name="research_agent",
    )
