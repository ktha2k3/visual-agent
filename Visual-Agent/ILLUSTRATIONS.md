# Legacy template illustrations

The current default is general scene composition, documented in
[GENERAL-SCENES.md](GENERAL-SCENES.md). The four-template implementation below is
retained for historical payloads/tests; it no longer limits new chat drawings.

Chat now uses validated JSON specifications and fixed SVG components. It does
not ask the model for executable Python/JavaScript and does not render a movie
for every question. Existing video history and the standalone Manim tool remain
available; the five-second Manim timeout is unchanged.

Supported templates:

- `function_graph`: a single real function of x, sampled on a bounded domain.
- `tangent`: graph and numerical tangent at the requested x0; unstable/cusp
  derivatives are rejected. The numerical approximation is labelled.
- `simple_pendulum`: SI length/gravity, initial amplitude up to 20 degrees,
  small-angle harmonic approximation without friction, labelled on screen.
- `harmonic_motion`: x(t)=A cos(omega*t+phase), with period and normalized graph.

The backend supplies sampled graph segments and calculated physics values. The
browser executes only trusted components: SVG, play/pause/reset and SVG download.
It respects reduced-motion preference on initial load. Expression evaluation uses
a bounded AST interpreter, never eval or model-generated code. Allowed functions:
sin/cos/tan/exp/log/sqrt/abs. Constant powers up to 12; no arbitrary names, imports,
attribute access or calls. Discontinuities split graph segments; entirely invisible
or invalid graphs are rejected.

The structured model call selects a template and parameters from the current
request, extracted problem and bounded conversation. It has a 15-second deadline
including structured-output fallback; no render/repair retry loop. The tutor stream
runs independently and can deliver text before the illustration. Unsupported,
invalid and timed-out requests produce explicit status without discarding text.
Late results are attached to the same assistant message and upserted into history.

Examples and assumptions are displayed. Schema/domain checks do not prove that
the model transcribed every datum correctly. Arbitrary 3D geometry, multiple
curves, arbitrary scenes, large-angle pendulums and additional physics systems
are intentionally unsupported until a tested template is added.

Verification: backend tests cover safe expressions, poles, tangent slopes/cusps,
physics parameters, timeouts, cancellation, independent text streaming, and late
success/failure persistence. Browser checks cover the four templates, simulation
controls, SVG download and unsupported output.
