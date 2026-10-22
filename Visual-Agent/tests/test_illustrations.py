import asyncio
import math

import pytest
from pydantic import ValidationError

from tools.visualization.templates import IllustrationSpec, TemplateIllustrationService, build_scene
from tools.visualization.streaming import illustration_stream


def test_graph_and_tangent_preserve_function_and_compute_points():
    scene = build_scene(IllustrationSpec(template="tangent", expression="x^2", x0=2))
    assert scene["tangent"]["slope"] == pytest.approx(4, abs=1e-5)
    assert scene["tangent"]["y"] == 4
    assert scene["expression"] == "x^2"
    assert scene["segments"]


@pytest.mark.parametrize("expression", ["__import__('os')", "x.__class__", "[x]", "x**999999", "sin(x, x)"])
def test_no_arbitrary_expression_execution(expression):
    with pytest.raises(ValueError):
        build_scene(IllustrationSpec(template="function_graph", expression=expression))


def test_graph_splits_at_singularity():
    scene = build_scene(IllustrationSpec(template="function_graph", expression="1/x", x_min=-2, x_max=2))
    assert len(scene["segments"]) >= 2
    assert all(not (segment[0][0] < 0 < segment[-1][0]) for segment in scene["segments"])


def test_high_frequency_graph_is_rejected_instead_of_aliasing_to_a_flat_line():
    with pytest.raises(ValueError, match="resolution"):
        build_scene(IllustrationSpec(template="function_graph", expression="sin(80*pi*x)", x_min=0, x_max=10))


def test_periodic_graph_refines_sampling_for_its_domain():
    scene = build_scene(IllustrationSpec(template="function_graph", expression="sin(2*pi*x)", x_min=-20, x_max=20))
    values = [y for segment in scene["segments"] for x, y in segment]
    assert max(values) > 0.9 and min(values) < -0.9


def test_physics_uses_validated_parameters_and_marks_defaults():
    scene = build_scene(IllustrationSpec(template="simple_pendulum", length_m=2, gravity=10, angle_deg=10))
    assert scene["period"] == pytest.approx(2 * math.pi * math.sqrt(0.2))
    assert scene["assumptions"]
    with pytest.raises(ValidationError):
        IllustrationSpec(template="simple_pendulum", length_m=-1)
    with pytest.raises(ValidationError):
        IllustrationSpec(template="harmonic_motion", omega=float("nan"))


@pytest.mark.asyncio
async def test_service_receives_full_grounded_question_and_never_codegen():
    class Provider:
        async def structured(self, schema, *, system, prompt):
            assert "y=x^3" in prompt and "vẽ tiếp tuyến" in prompt
            return IllustrationSpec(template="tangent", expression="x^3", x0=1)

    result = await TemplateIllustrationService(provider=Provider()).visualize(
        question="Đề ảnh: y=x^3. Yêu cầu: vẽ tiếp tuyến", subject="math", user_id="a"
    )
    assert result.success and result.scene_data["tangent"]["slope"] == pytest.approx(3, abs=1e-5)
    assert result.video_url is None


@pytest.mark.asyncio
async def test_unsupported_and_timeout_have_explicit_status():
    class Provider:
        async def structured(self, *args, **kwargs):
            return IllustrationSpec(template="unsupported", reason="Chưa có mẫu hình không gian")
    result = await TemplateIllustrationService(provider=Provider()).visualize(question="hình chóp", subject="math", user_id="a")
    assert not result.success and result.status == "unsupported"
    assert result.error


@pytest.mark.asyncio
async def test_slow_model_times_out_without_retry_or_render():
    class Provider:
        async def structured(self, *args, **kwargs):
            await asyncio.sleep(10)
    result = await TemplateIllustrationService(provider=Provider(), timeout=0.01).visualize(
        question="con lắc", subject="physics", user_id="a")
    assert not result.success and result.status == "failed"


def test_cusp_cannot_be_reported_as_a_tangent():
    with pytest.raises(ValueError):
        build_scene(IllustrationSpec(template="tangent", expression="abs(x)", x0=0))


def test_harmonic_period_and_phase_are_preserved():
    scene = build_scene(IllustrationSpec(template="harmonic_motion", amplitude=2, omega=math.pi, phase=0.5))
    assert scene["period"] == 2
    assert scene["amplitude"] == 2 and scene["phase"] == 0.5


def test_grounding_keeps_image_problem_and_current_request_under_input_limit():
    import json
    from tools.visualization.templates import illustration_question
    packet = illustration_question("Đề ảnh: l=2m " + "a" * 12000, "minh hoạ bài này", [{"role": "assistant", "content": "b" * 8000}] * 24)
    data = json.loads(packet)
    assert data["current_request"] == "minh hoạ bài này"
    assert data["problem"].startswith("Đề ảnh: l=2m")
    assert len(data["conversation"]) == 6
    assert len(packet) < 18000


@pytest.mark.parametrize("question", ["Vẽ đồ thị y=x^2", "Cho hình minh hoạ bài này", "Mo phong con lac"])
def test_explicit_illustration_requests_are_routed(question):
    from tools.visualization.routing import decide_visualization
    assert decide_visualization(question, "math").should_visualize


@pytest.mark.asyncio
async def test_cancelling_stream_cleans_up_illustration():
    started, stopped = asyncio.Event(), asyncio.Event()
    class Graph:
        async def astream(self, *args, **kwargs):
            yield "updates", {"visualization_router": {"visualization_decision": {"should_visualize": True, "subject": "math", "question": "x^2"}}}
            await started.wait()
    class Service:
        async def visualize(self, **kwargs):
            started.set()
            try:
                await asyncio.sleep(100)
            finally:
                stopped.set()
    iterator = illustration_stream(Graph(), {"user_id": "a"}, {}, service=Service())
    await anext(iterator)
    await asyncio.wait_for(started.wait(), 1)
    await iterator.aclose()
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_fresh_physics_request_replaces_previous_problem_without_equals(monkeypatch):
    from education.models import ProblemExtraction, Subject
    from education.tutor_graph import build_tutor_graph
    from services.repository import MemoryRepository
    async def extract(self, *, message, images):
        return ProblemExtraction(subject=Subject.PHYSICS, problem_markdown=message, confidence=0.95)
    monkeypatch.setattr("education.tutor_graph.ProblemPerception.extract", extract)
    graph = build_tutor_graph(MemoryRepository(), object(), research_service=object())
    previous = ProblemExtraction(subject=Subject.PHYSICS, problem_markdown="Con lắc đơn dài 1m", confidence=0.95)
    message = "Mô phỏng dao động điều hòa biên độ 5 cm, chu kỳ 2 s"
    result = await graph.builder.nodes["perception"].runnable.ainvoke({
        "user_id": "student", "session_id": "fresh", "subject": "physics",
        "message": message, "problem": previous.model_dump(), "attachment_ids": [],
    })
    assert result["problem"]["problem_markdown"] == message
    followup = await graph.builder.nodes["perception"].runnable.ainvoke({
        "user_id": "student", "session_id": "fresh", "subject": "physics",
        "message": "Mô phỏng bài này rõ hơn", "problem": previous.model_dump(), "attachment_ids": [],
    })
    assert "problem" not in followup  # retain the extracted prior problem
    answer = await graph.builder.nodes["perception"].runnable.ainvoke({
        "user_id": "student", "session_id": "fresh", "subject": "physics",
        "message": "x=2", "problem": previous.model_dump(), "attachment_ids": [],
    })
    assert "problem" not in answer  # a learner answer is not a new problem


@pytest.mark.asyncio
async def test_text_stream_does_not_wait_for_illustration_and_late_result_is_delivered():
    gate = asyncio.Event()
    class Graph:
        async def astream(self, *args, **kwargs):
            yield "updates", {"visualization_router": {"visualization_decision": {"should_visualize": True, "subject": "math", "question": "y=x^2"}}}
            yield "custom", {"type": "token", "content": "Lời giải"}
    class Service:
        async def visualize(self, **kwargs):
            await gate.wait()
            return await TemplateIllustrationService(provider=Provider()).visualize(**kwargs)
    class Provider:
        async def structured(self, *args, **kwargs):
            return IllustrationSpec(template="function_graph", expression="x^2")
    iterator = illustration_stream(Graph(), {"user_id": "a"}, {}, service=Service())
    assert (await anext(iterator))[1].get("visualization_router")
    event = await asyncio.wait_for(anext(iterator), 1)
    assert event[1]["type"] == "token"
    gate.set()
    remaining = [event async for event in iterator]
    result = next(payload["visualization_tool"]["visualization"] for mode, payload in remaining if mode == "updates" and "visualization_tool" in payload)
    assert result["success"] and result["scene_data"]
