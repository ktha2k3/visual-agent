from pathlib import Path

import yaml

from tools.visualization.renderer import ManimRenderer


def test_opengl_command_writes_video_without_preview(tmp_path):
    command = ManimRenderer(renderer="opengl").build_command(tmp_path, "Demo")
    assert command[command.index("--renderer") + 1] == "opengl"
    assert "--write_to_movie" in command
    assert "--preview" not in command


def test_gpu_overlay_allocates_gpu_and_selects_opengl():
    config = yaml.safe_load(Path("docker-compose.gpu.yml").read_text())
    backend = config["services"]["backend"]
    assert backend["gpus"] == "all"
    assert backend["environment"]["VISUALIZATION_RENDERER"] == "opengl"
