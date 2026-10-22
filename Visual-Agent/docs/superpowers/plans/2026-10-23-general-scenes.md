# General scene implementation plan

Approved design: replace four-topic selection with a declarative scene composed
from objects, relations, expressions, animations and labels. Preserve independent
text streaming and existing historical illustrations. No generated Python/JS.

Implementation: bounded Pydantic scene schema and safe numeric expression compiler;
SVG for arbitrary 2D compositions and Three.js for 3D compositions. Model can
compose geometry instead of selecting a topic. Derived geometry and animation
frames are computed by trusted backend code. Three.js is loaded only for 3D.

- [x] Test scene composition, derived relations, expression validation, unknown references/cycles, resource bounds and 3D.
- [x] Implement generic scene compiler, schema and provider prompt; use as default chat service.
- [x] Render composed 2D/3D scenes with controls, labels and exports; preserve history compatibility.
- [x] Verify with backend suite, browser tests (projectile, optics, free diagrams, 3D), build and real model smoke.
- [x] Review correctness and update docs; reload running Docker backend/frontend.

Limits: no guarantee of arbitrary scientific accuracy or photorealistic anatomy.
No external image URLs or executable scene scripts. Reject unresolved/invalid
relationships instead of guessing coordinates. Preserve input evidence/assumptions.
Time-based expressions are sampled with bounded resolution and interpolated by
the browser. Keep the existing independent chat stream and Manim timeout.

Verification: 129 backend tests, 2 frontend numeric tests, production build, Edge browser geometry/optics/projectile/3D/export/mobile checks, live-model optics and cube. Review fixes: temporal resolution, relation source validation, camera-relative infinite lines and planar 3D polygons. Runtime default SceneIllustrationService; health200.
