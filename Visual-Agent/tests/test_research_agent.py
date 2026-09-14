from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

import agents.research_agent as research_module
from agents.research_agent import RESEARCH_TOOLS, RelatedProblemResearcher
from tools.research.routing import should_research_problem


def test_research_agent_exposes_expected_research_tools():
    assert {tool.name for tool in RESEARCH_TOOLS} == {
        "tavily",
        "arxiv",
        "wikipedia",
        "current_datetime",
    }


def test_research_routing_selects_solution_and_deep_concept_requests():
    assert should_research_problem("Hãy tính đạo hàm của x^3")
    assert should_research_problem("Giúp mình giải câu này")
    assert should_research_problem("Tìm một bài toán tương tự")
    assert should_research_problem("Giải thích định luật II Newton")
    assert should_research_problem("Giúp mình hiểu sâu bản chất của đạo hàm")
    assert not should_research_problem("Em tính được kết quả là 12")
    assert not should_research_problem("Mình không nhớ")


@pytest.mark.asyncio
async def test_research_agent_receives_grounded_problem_and_returns_report(monkeypatch):
    captured = {}

    class FakeAgent:
        async def ainvoke(self, payload, config):
            captured["prompt"] = payload["messages"][0].content
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Kiến thức: dùng quy tắc đạo hàm.\n"
                            "Nguồn: https://example.edu/calculus"
                        )
                    )
                ]
            }

    monkeypatch.setattr(research_module, "research_agent", FakeAgent())
    monkeypatch.setattr(research_module, "concept_research_agent", FakeAgent())
    report = await RelatedProblemResearcher().research(
        question="Hãy tính đạo hàm",
        subject="math",
        problem="Tính đạo hàm của x^3",
        recent_conversation=[{"role": "user", "content": "Em chưa biết bắt đầu"}],
        research_task="Tìm dạng bài cơ bản đến nâng cao về đạo hàm",
        learning_intent="understand_concept",
        understanding_level="developing",
    )

    assert "https://example.edu/calculus" in report
    assert "Tính đạo hàm của x^3" in captured["prompt"]
    assert "Tìm dạng bài cơ bản đến nâng cao" in captured["prompt"]
    assert "understand_concept" in captured["prompt"]
    assert captured["config"]["recursion_limit"] == 12


@pytest.mark.asyncio
async def test_research_service_routes_to_agent_for_explicit_mode(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, payload, config):
            calls.append(self.name)
            return {"messages": [AIMessage(content=self.name)]}

    monkeypatch.setattr(research_module, "research_agent", FakeAgent("guided"))
    monkeypatch.setattr(
        research_module, "concept_research_agent", FakeAgent("concept")
    )
    researcher = RelatedProblemResearcher()

    guided = await researcher.research(
        question="Giải bài",
        subject="math",
        problem="Bài đạo hàm",
        recent_conversation=[],
        learning_intent="solve_problem",
    )
    concept = await researcher.research(
        question="Đạo hàm là gì?",
        subject="math",
        problem="Khái niệm đạo hàm",
        recent_conversation=[],
        learning_intent="understand_concept",
    )

    assert guided == "guided"
    assert concept == "concept"
    assert calls == ["guided", "concept"]
