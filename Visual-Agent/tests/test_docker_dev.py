from pathlib import Path

import yaml
from uvicorn import Config
from uvicorn.supervisors.watchfilesreload import FileFilter


def test_dev_reloader_ignores_generated_manim_python_files():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    command = compose["services"]["backend"]["command"]
    excludes = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--reload-exclude"
    ]
    file_filter = FileFilter(
        Config("api:app", reload=True, reload_excludes=excludes)
    )

    assert not file_filter(
        Path("/app/data/visualizations/artifact-id/stem_scene.py")
    )
    assert file_filter(Path("/app/api.py"))
