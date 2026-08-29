from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor

from agents.research_agent import research_agent
from agents.vision_agent import vision_agent
from utils.config import get_settings

_PROMPT = (
    "You are a supervisor managing two specialized agents:\n"
    "- research_agent: evaluates approved exam documents from the two latest completed exam years\n"
    "- vision_agent: handles visual tasks such as describing images and detecting objects\n\n"
    "INSTRUCTIONS:\n"
    "- Route each user request to the appropriate agent based on the task type\n"
    "- For exam distribution and score-band planning, delegate the approved research document to research_agent\n"
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
    settings = get_settings()
    return create_supervisor(
        model=init_chat_model(
            model=settings.tutor_model,
            model_provider="openai",
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
        ),
        agents=[research_agent, vision_agent],
        prompt=_PROMPT,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(checkpointer=checkpointer)
