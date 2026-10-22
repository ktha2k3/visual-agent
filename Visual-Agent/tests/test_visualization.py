from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver

from education.models import ProblemExtraction, Subject
from education.tutor_graph import TutorAssessment, build_tutor_graph
from services.repository import MemoryRepository
from tools.visualization.artifacts import VisualizationArtifactStore
from tools.visualization.models import (
    AnimationBeat,
    ConceptAnalysis,
    RenderResult,
    StemVisualizationResult,
    VisualObject,
    VisualizationPlan,
)
from tools.visualization.provider import LLMProvider, OllamaProvider
from tools.visualization.renderer import RenderError
from tools.visualization.renderer import ManimRenderer
from tools.visualization.routing import decide_visualization
from tools.visualization.service import StemVisualizationService
from tools.visualization.tool import visualize_stem_concept
from tools.visualization.validation import extract_python_code, validate_manim_code
from utils.config import get_settings


VALID_SCENE = """from manim import *

class StemScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait(0.5)
"""


def concept(subject: str = "math") -> ConceptAnalysis:
    return ConceptAnalysis(
        subject=subject,
        concept="Đạo hàm",
        explanation="Đạo hàm mô tả tốc độ thay đổi tức thời của một đại lượng.",
        formulas=[r"f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}"],
        prerequisites=["Hàm số", "Hệ số góc"],
        key_takeaway="Cát tuyến tiến tới tiếp tuyến khi hai điểm tiến gần nhau.",
    )


def plan() -> VisualizationPlan:
    return VisualizationPlan(
        concept="Đạo hàm",
        learning_goal="Nhìn thấy hệ số góc cát tuyến hội tụ về hệ số góc tiếp tuyến.",
        objects=[
            VisualObject(name="graph", kind="graph", purpose="Đồ thị hàm số"),
            VisualObject(name="secant", kind="geometry", purpose="Đường cát tuyến"),
        ],
        relationships=["Hai điểm gần nhau làm cát tuyến tiến tới tiếp tuyến"],
        variable_changes=["h tiến về 0"],
        camera_movements=["Giữ đồ thị trong frame"],
        transformations=["cát tuyến thành tiếp tuyến"],
        timeline=[
            AnimationBeat(order=1, action="Vẽ đồ thị và hai điểm", narration="Chọn hai điểm trên đồ thị"),
            AnimationBeat(order=2, action="Cho hai điểm tiến gần nhau", narration="Hệ số góc hội tụ"),
        ],
    )


class FakeProvider(LLMProvider):
    def __init__(self, codes: list[str] | None = None):
        self.codes = list(codes or [VALID_SCENE])
        self.structured_calls = 0
        self.text_calls = 0

    async def structured(self, schema, *, system, prompt):
        self.structured_calls += 1
        return concept() if schema is ConceptAnalysis else plan()

    async def text(self, *, system, prompt):
        self.text_calls += 1
        return f"```python\n{self.codes.pop(0)}\n```"


class FakeRenderer:
    def __init__(self, failures: int = 0):
        self.failures = failures
        self.calls = 0

    async def render(self, run_dir: Path, scene_name: str) -> RenderResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise RenderError("simulated Manim compile failure")
        video = run_dir / "media" / "videos" / "stem.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"0" * 2048)
        return RenderResult(
            success=True,
            video_relative_path=video.relative_to(run_dir).as_posix(),
            duration_seconds=0.01,
        )


def service(tmp_path, provider=None, renderer=None, retries=3):
    settings = replace(
        get_settings(),
        visualization_artifact_dir=str(tmp_path),
        visualization_max_render_retries=retries,
    )
    return StemVisualizationService(
        provider=provider or FakeProvider(),
        renderer=renderer or FakeRenderer(),
        store=VisualizationArtifactStore(tmp_path),
        settings=settings,
    )


@pytest.mark.parametrize(
    ("question", "subject"),
    [
        ("Đạo hàm là gì?", Subject.MATH),
        ("Giải thích tích phân.", Subject.MATH),
        ("Giải thích định lý Pythagore.", Subject.MATH),
        ("Giải thích vector.", Subject.MATH),
        ("Giải thích eigenvalue.", Subject.MATH),
        ("Giải thích F = ma.", Subject.PHYSICS),
        ("Giải thích chuyển động ném xiên.", Subject.PHYSICS),
        ("Giải thích dao động điều hòa.", Subject.PHYSICS),
        ("Giải thích bảo toàn năng lượng.", Subject.PHYSICS),
    ],
)
def test_concept_questions_always_route_to_visualization(question, subject):
    decision = decide_visualization(question, subject)
    assert decision.should_visualize is True
    assert decision.subject == subject.value


def test_plain_calculation_also_forces_visualization():
    assert decide_visualization("Tính 2 + 2", Subject.MATH).should_visualize
    assert decide_visualization("Giải phương trình x + 1 = 3", Subject.MATH).should_visualize
    assert not decide_visualization("Giải thích phản ứng este", Subject.CHEMISTRY).should_visualize


def test_codegen_parsing_and_security_validation():
    assert validate_manim_code(extract_python_code(f"```python\n{VALID_SCENE}```")).valid
    unsafe = "from manim import *\nimport os\nclass Bad(Scene):\n def construct(self):\n  self.play(Create(Circle()))\n"
    report = validate_manim_code(unsafe)
    assert not report.valid
    assert "Blocked import: os" in report.errors
    missing_font = validate_manim_code(
        VALID_SCENE.replace("Circle()", "Text('Đạo hàm')"),
        required_text_font="Arial",
    )
    assert not missing_font.valid
    assert any("Vietnamese Text" in error for error in missing_font.errors)
    invalid_easing = validate_manim_code(
        VALID_SCENE.replace("self.wait(0.5)", "self.wait(0.5)\n        self.play(Create(Square()), rate_func=ease_in_quad)")
    )
    assert not invalid_easing.valid
    assert any("rate_functions.ease_in_quad" in error for error in invalid_easing.errors)
    wrong_keyword = validate_manim_code(
        VALID_SCENE.replace(
            "self.wait(0.5)",
            "self.wait(0.5)\n        self.play(Create(Square()), rate_functions=rate_functions.smooth)",
        )
    )
    assert not wrong_keyword.valid
    assert any("rate_func` keyword" in error for error in wrong_keyword.errors)
    invalid_text_size = validate_manim_code(
        VALID_SCENE.replace("Circle()", "Text('label', size=0.5)")
    )
    assert not invalid_text_size.valid
    assert any("font_size" in error for error in invalid_text_size.errors)
    font_constant = validate_manim_code(
        """from manim import *
font_name = 'Arial'
class Demo(Scene):
    def construct(self):
        label = Text('\u0110\u1ecbnh lu\u1eadt', font=font_name)
        self.play(Write(label))
        self.wait(0.5)
""",
        required_text_font="Arial",
    )
    assert font_constant.valid


@pytest.mark.asyncio
async def test_ollama_provider_falls_back_to_json_schema_parsing():
    class BrokenStructured:
        async def ainvoke(self, messages):
            raise RuntimeError("tool calls unsupported")

    class JsonModel:
        def with_structured_output(self, schema, **kwargs):
            return BrokenStructured()

        async def ainvoke(self, messages):
            return AIMessage(content=f"```json\n{concept().model_dump_json()}\n```")

    parsed = await OllamaProvider(model=JsonModel()).structured(
        ConceptAnalysis, system="system", prompt="question"
    )
    assert parsed.concept == "Đạo hàm"
    assert parsed.formulas


@pytest.mark.asyncio
async def test_pipeline_parses_plans_renders_and_reuses_cache(tmp_path):
    provider = FakeProvider()
    renderer = FakeRenderer()
    pipeline = service(tmp_path, provider=provider, renderer=renderer)
    first = await pipeline.visualize(
        question="Đạo hàm là gì?", subject="math", user_id="student-a"
    )
    second = await pipeline.visualize(
        question="  đạo HÀM   là gì? ", subject="math", user_id="student-b"
    )
    assert first.success and first.video_url
    assert second.success and second.cached
    assert first.artifact_id == second.artifact_id
    assert provider.structured_calls == 2
    assert provider.text_calls == 1
    assert renderer.calls == 1
    store = VisualizationArtifactStore(tmp_path)
    assert store.resolve_owned_file(first.artifact_id, "student-a", kind="video")
    assert store.resolve_owned_file(first.artifact_id, "student-b", kind="video")
    assert store.resolve_owned_file(first.artifact_id, "intruder", kind="video") is None


@pytest.mark.asyncio
async def test_validation_failure_is_repaired_before_render(tmp_path):
    invalid = "from manim import *\nclass Broken(Scene)\n    pass"
    provider = FakeProvider(codes=[invalid, VALID_SCENE])
    renderer = FakeRenderer()
    result = await service(
        tmp_path, provider=provider, renderer=renderer, retries=2
    ).visualize(question="Giải thích vector", subject="math", user_id="student")
    assert result.success
    assert result.attempts == 2
    assert provider.text_calls == 2
    assert renderer.calls == 1


@pytest.mark.asyncio
async def test_invalid_updater_is_repaired_without_rendering_broken_scene(tmp_path):
    invalid = VALID_SCENE.replace(
        "self.play(Create(circle))",
        "circle.add_updater(lambda: None)\n        self.play(Create(circle))",
    )
    provider = FakeProvider(codes=[invalid, VALID_SCENE])
    renderer = FakeRenderer()
    result = await service(
        tmp_path, provider=provider, renderer=renderer, retries=1
    ).visualize(question="Giải thích con lắc", subject="physics", user_id="student")
    assert result.success
    assert result.attempts == 2
    assert renderer.calls == 1
    import json

    first_validation = json.loads(
        (tmp_path / result.artifact_id / "validation-1.json").read_text()
    )
    assert not first_validation["valid"]
    assert "add_updater" in first_validation["errors"][0]


@pytest.mark.asyncio
async def test_render_failure_triggers_bounded_self_repair(tmp_path):
    provider = FakeProvider(codes=[VALID_SCENE, VALID_SCENE])
    renderer = FakeRenderer(failures=1)
    result = await service(
        tmp_path, provider=provider, renderer=renderer, retries=1
    ).visualize(question="Giải thích F = ma", subject="physics", user_id="student")
    assert result.success
    assert result.attempts == 2
    assert renderer.calls == 2
    assert provider.text_calls == 2


@pytest.mark.asyncio
async def test_tool_interface_invokes_pipeline(tmp_path):
    pipeline = service(tmp_path)
    result = await visualize_stem_concept(
        "Đạo hàm là gì?", "math", user_id="student", service=pipeline
    )
    assert result.success
    assert result.concept == "Đạo hàm"


@pytest.mark.asyncio
async def test_real_manim_renderer_smoke(tmp_path):
    source = Path("tests/fixtures/stem_smoke_scene.py").read_text(encoding="utf-8")
    (tmp_path / "stem_scene.py").write_text(source, encoding="utf-8")
    result = await ManimRenderer(quality="l", timeout=30).render(
        tmp_path, "StemSmokeScene"
    )
    video = tmp_path / result.video_relative_path
    assert result.success
    assert video.is_file()
    assert video.stat().st_size >= 1024


class _Storage:
    async def read(self, record):  # pragma: no cover
        return b""


@pytest.mark.asyncio
async def test_agent_falls_back_to_text_when_visualization_fails(monkeypatch):
    async def fake_extract(self, *, message, images):
        return ProblemExtraction(
            subject=Subject.MATH,
            problem_markdown=message,
            skill_ids=["math.derivatives"],
            confidence=0.95,
        )

    class FailedVisualizer:
        calls = 0

        async def visualize(self, **kwargs):
            self.calls += 1
            return StemVisualizationResult(
                success=False,
                subject="math",
                concept="Đạo hàm",
                explanation="Giải thích bằng văn bản",
                key_takeaway="Đạo hàm là tốc độ thay đổi tức thời.",
                error="render unavailable",
            )

    class FakeAssessment:
        async def ainvoke(self, messages):
            return TutorAssessment()

    class StreamingModel:
        async def astream(self, messages):
            yield AIMessageChunk(content="Đạo hàm mô tả tốc độ thay đổi tức thời.")

        def with_structured_output(self, schema, **kwargs):
            return FakeAssessment()

    class FakeEmbedding:
        async def aembed_query(self, query):
            return [0.0] * 768

    visualizer = FailedVisualizer()
    monkeypatch.setattr("education.tutor_graph.ProblemPerception.extract", fake_extract)
    monkeypatch.setattr("education.tutor_graph.get_chat_model", lambda role: StreamingModel())
    monkeypatch.setattr("education.tutor_graph.get_embedding_model", lambda: FakeEmbedding())
    graph = build_tutor_graph(
        MemoryRepository(),
        _Storage(),
        checkpointer=MemorySaver(),
        visualization_service=visualizer,
    )
    result = await graph.ainvoke(
        {
            "user_id": "student",
            "session_id": "session",
            "message": "Giải thích đạo hàm là gì?",
            "attachment_ids": [],
            "conversation": [{"role": "user", "content": "Giải thích đạo hàm là gì?"}],
        },
        {"configurable": {"thread_id": "student:visualization-fallback"}},
    )
    assert visualizer.calls == 1
    assert not result["visualization"]["success"]
    assert "tốc độ thay đổi" in result["tutor_response"]["message"]
