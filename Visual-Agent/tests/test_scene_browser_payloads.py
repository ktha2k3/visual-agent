import json
from pathlib import Path

from tools.visualization.scene import SceneSpec, compile_scene


def test_composed_scene_fixtures_compile_without_topic_templates():
    specs = json.loads(Path("tests/fixtures/composed_scenes.json").read_text())
    scenes = [compile_scene(SceneSpec.model_validate(spec)) for spec in specs]
    assert all(scene["template"] == "scene" for scene in scenes)
    assert scenes[2]["frames"]
    assert scenes[3]["dimension"] == 3
