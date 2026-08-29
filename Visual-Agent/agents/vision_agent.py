try:
    from langchain.agents import create_agent
except ImportError:  # LangChain 0.3 compatibility while migrating the lockfile
    from langgraph.prebuilt import create_react_agent as create_agent

from tools.vision.describe_tool import image_describer_tool
from tools.vision.detect_tool import detect_and_count_tool
from utils.model_factory import get_chat_model

_PROMPT = (
        "You are a vision agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with visual tasks (e.g., describing images, detecting and counting objects)\n"
        "- Use only the tools provided to analyze visual inputs\n"
        "- After completing your task, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
)

try:
    vision_agent = create_agent(
        model=get_chat_model("vision"),
        tools=[image_describer_tool, detect_and_count_tool],
        system_prompt=_PROMPT,
        name="vision_agent",
    )
except TypeError:
    vision_agent = create_agent(
        model=get_chat_model("vision"),
        tools=[image_describer_tool, detect_and_count_tool],
        prompt=_PROMPT,
        name="vision_agent",
    )
