"""Verify the GPU context and compare a small Cairo/OpenGL render offline.

Run inside the backend: python -m scripts.check_gpu_render
No model calls; benchmark artifacts are temporary.
"""
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import moderngl

from tools.visualization.renderer import ManimRenderer


async def main():
    context = moderngl.create_context(standalone=True, backend="egl")
    renderer = context.info["GL_RENDERER"]
    print("GL_RENDERER:", renderer, flush=True)
    if any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "swrast")):
        raise RuntimeError("OpenGL is using software rendering, not the GPU")
    context.release()
    failures = []
    for backend in ("cairo", "opengl"):
        with tempfile.TemporaryDirectory(prefix=f"manim-{backend}-") as directory:
            root = Path(directory)
            source = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/stem_smoke_scene.py"
            scene_name = sys.argv[2] if len(sys.argv) > 2 else "StemSmokeScene"
            shutil.copyfile(source, root / "stem_scene.py")
            try:
                result = await ManimRenderer(renderer=backend, timeout=5).render(root, scene_name)
                print(backend, result.model_dump(), flush=True)
            except Exception as exc:
                print(backend, type(exc).__name__, str(exc)[-1500:], flush=True)
                failures.append(backend)
    if failures:
        raise SystemExit("Render checks failed: " + ", ".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
