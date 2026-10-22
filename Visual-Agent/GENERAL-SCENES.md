# General scientific scene composition

Default chat illustrations now use `SceneIllustrationService`, not four topic
templates. The model supplies a bounded declarative scene; trusted backend code
validates and compiles it. No generated Python, JavaScript, HTML or library API
calls are executed. Existing four-template history and video playback still work.

## Composable objects

Points, lines, segments, arrows, circles, ellipses, polygons, polylines, parametric
curves, labels and axes compose new 2D diagrams. Spheres, boxes, cylinders, curves,
arrows and planar polygons also compose 3D scenes. SVG handles 2D; lazily loaded
Three.js handles 3D with orbit/zoom controls. No topic registry is required for a
new diagram made from these objects. Examples in `tests/fixtures/composed_scenes.json`
cover geometry, thin-lens optics, projectile motion and a schematic 3D molecule.

`parameters` store values and units. Coordinates may reference named parameters,
time `t`, curve parameter `u`, or another object's anchor via `id_x/id_y/id_z`.
In `points` only, a standalone object ID also selects the coordinate slot's axis
(e.g. `["anchor", 0]` means `["anchor_x", 0]`). Existing parameter/variable names
take precedence. Arithmetic and scalar fields still require explicit axis names;
unknown IDs and dependency cycles remain errors. This local normalization avoids
an extra model call for a coordinate shorthand.
Zero arrows are valid (for example, velocity at a turning point) and render
without an arrowhead. Infinite lines and direction-defining relation sources
still require distinct points.
`relations` compute midpoint, line/segment intersection, parallel and perpendicular
constructions. References are dependency-sorted; cycles, missing anchors,
degenerate lines, nonplanar polygons and unresolved relations are rejected.
Relations construct target objects; they do not assert constraints between
already specified lines. In 3D, midpoint and parallel are supported; other
geometry can use explicit coordinates. The model schema and prompt state this
distinction, including that a target must not appear among its own sources.

Every supported Math/Physics turn requests an independent illustration,
including calculations, short answers and turns requiring clarification.
No drawing keywords or intent filter is required. This application-wide policy
also applies to messages asking for text alone. Missing essential data can still
produce an explicit unsupported result rather than fabricated geometry.

Math expressions use a bounded AST interpreter for numeric arithmetic and an
allowlist of math functions. Arbitrary calls/attributes, imports and nonnumeric
values are forbidden. Powers are constant and bounded. Discontinuous curves split
into segments. Trig variation is checked against spatial and temporal sampling
resolution; underresolved scenes fail rather than displaying a false static line.

## Time and lifecycle

Animated coordinates are sampled by the backend and interpolated numerically in
the browser. Play/pause/reset, scrubbing, playback speed and initial reduced-motion
preference are supported. Scenes have at most 60 objects, 30 parameters, 40
relations, 4,096 samples per curve, 721 time frames and a 60,000-vertex compilation
budget (also capped at 6,000 object/frame instances and 500,000 work units).
Curve `samples` is an initial hint. Generic interval bounds, local chord error and
trigonometric phase variation refine curves and time intervals to a nominal
half-pixel tolerance at a 640x400 reference viewport. Animated curves share
parameter knots across frames. This changes sampling density, never the requested
formula or frequency. Unresolved features produce an explicit resource error.
Motion is a sampled approximation, not an arbitrary differential-equation
solver. 3D resources, labels, controls and render loops are disposed on unmount.

Text streams independently. Model generation, one optional data-only validation
repair and compilation share a 20-second request deadline. There is no render-code
repair loop. Failures and missing data produce explicit status; late results are
attached to the same assistant message and persisted in history. The standalone
Manim video tool and its five-second render timeout are unchanged.

Compilation checks the shared deadline even inside its worker thread. Sampling
budget errors do not invoke model repair. Failed attempts save private JSON under
`data/visualizations/scene-errors/<diagnostic_id>-<attempt>.json`, including the
scene, bounded raw response and error. Normal logs contain only the diagnostic ID,
attempt and error type. The attempt count includes a failed repair call.

2D scenes export SVG. 3D scenes export PNG at the current view; no server-side GPU
or MP4 encoding is involved in default chat. WebGL availability is reported on the
client rather than silently claiming a successful display.

## Scope and evidence

This is a general composition vocabulary, not a promise to correctly illustrate
every science topic. Numeric/schema validation does not establish scientific truth
or guarantee that model transcription is correct. Approximation assumptions and
example values are shown to the learner. Model-generated layouts can still need
better label placement or a clearer choice of viewport.

Photorealistic anatomy, sourced reference images and generative bitmap illustration
are not implemented in this scene compiler. Requests that require such evidence
must return a reason instead of fabricating a detailed anatomical image. Existing
public subject selection remains Math/Physics; this change expands drawing
capability rather than changing the subject/domain policy of the rest of the app.

Verified with backend regressions, frontend numeric interpolation tests, production
build, headless Edge (2D/3D, controls, downloads, mobile), and real-model thin-lens
and cube output. The React/Vite chunk-size warning is advisory; Three.js is loaded
only for 3D scenes.
