import os

from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor

from agents.research_agent import research_agent
from agents.vision_agent import vision_agent

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY",  "ollama")

_PROMPT = (
    "You are a supervisor managing two specialized agents:\n"
    "- research_agent: handles research tasks using arxiv and wikipedia\n"
    "- vision_agent: handles visual tasks such as describing images and detecting objects\n\n"
    "INSTRUCTIONS:\n"
    "- Route each user request to the appropriate agent based on the task type\n"
    "- For research or information lookup tasks, delegate to research_agent\n"
    "- For image analysis or object detection tasks, delegate to vision_agent\n"
    "- Synthesize the agent's response and return a clear final answer to the user\n"
    "- You have access to the full conversation history — use it to answer follow-up questions\n"
    "RULES:\n"
    "- Never make up information or guess\n"
    "- Do NOT perform any tasks yourself, always delegate to the appropriate agent\n"
    "- If the user's request is unclear, ask a short clarification question\n"
    "- Respond in the same language as the user"
)


def build_supervisor(checkpointer=None):
    """
    Compile the supervisor graph.
    Pass a checkpointer (AsyncRedisSaver or MemorySaver) for conversation memory.
    Called once at startup from api.py lifespan.
    """
    return create_supervisor(
        model=init_chat_model(
            model="kimi-k2.5:cloud",
            model_provider="openai",
            base_url=_OLLAMA_BASE_URL,
            api_key=_OLLAMA_API_KEY,
        ),
        agents=[research_agent, vision_agent],
        prompt=_PROMPT,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(checkpointer=checkpointer)
