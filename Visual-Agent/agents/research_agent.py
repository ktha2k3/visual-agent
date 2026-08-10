import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.research.search_tools import arxiv, wikipedia

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

research_agent = create_react_agent(
    model=ChatOpenAI(
        model="gpt-oss:120b-cloud",
        base_url=_OLLAMA_BASE_URL,
        api_key=_OLLAMA_API_KEY,
    ),
    tools=[arxiv, wikipedia],
    prompt=(
        "You are a research assistant. Use the provided tools to find information and research papers.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks. DO NOT do anything else.\n"
        "- Use tools to look up information; do not guess or fabricate facts.\n"
        "- After completing your task, respond to the supervisor directly.\n"
        "- Respond ONLY with the results of your work, do NOT include any other text."
    ),
    name="research_agent",
)
