# GPU rendering on this Windows host

The base Compose configuration uses Cairo on CPU. For Docker Desktop with WSL2,
NVIDIA GPU passthrough and WSLg, start the backend with:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build --no-deps backend
docker compose exec -T backend python -m scripts.check_gpu_render
```

Use both Compose files on subsequent backend recreations to keep GPU settings.
The overlay allocates the NVIDIA GPU and exposes WSL's Direct3D libraries/device
and WSLg socket. Mesa translates OpenGL to D3D12 on the selected NVIDIA adapter.
The image includes EGL/Mesa; Manim explicitly uses OpenGL and writes a movie
without opening a preview window. This overlay is specific to Windows WSL2,
not a generic Linux NVIDIA EGL configuration.

The check prints GL_RENDERER (must name the hardware GPU, not llvmpipe), then
renders the same small scene using Cairo and OpenGL. Artifacts are temporary.
It makes no model calls. The application's five-second render timeout remains
in effect; GPU rendering does not guarantee every scene finishes within it.
Python scene updates, LaTeX, text preparation and video encoding can still use CPU.

The overlay uses a separate visualization cache version to avoid reusing Cairo
results. To return to the base CPU configuration:

```powershell
docker compose -f docker-compose.yml up -d --no-deps backend
```

Verified on RTX 3060 Ti: GL_RENDERER reports `D3D12 (NVIDIA GeForce RTX 3060 Ti)`.
The 0.3-second smoke scene produced MP4 with both renderers: approximately
1.61 seconds Cairo and 1.98 seconds OpenGL (including process startup).
This small example does not demonstrate a speedup. The archived pendulum scene
timed out under Cairo at five seconds; OpenGL reported an incompatible
`Arc.set_angle` call. Existing generated scenes may need regeneration for OpenGL.
