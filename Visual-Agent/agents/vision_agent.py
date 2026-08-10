import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.vision.describe_tool import image_describer_tool
from tools.vision.detect_tool import detect_and_count_tool

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

vision_agent = create_react_agent(
    model=ChatOpenAI(
        model="gemma4:31b-cloud",
        base_url=_OLLAMA_BASE_URL,
        api_key=_OLLAMA_API_KEY,
    ),
    tools=[image_describer_tool, detect_and_count_tool],
    prompt=(
        "You are a vision agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with visual tasks (e.g., describing images, detecting and counting objects)\n"
        "- Use only the tools provided to analyze visual inputs\n"
        "- After completing your task, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="vision_agent",
)
