# Template illustrations implementation plan

Goal: implement the five priorities approved in chat: four fixed illustration templates,
grounded input, independent text streaming, truthful status, unsupported fallback.
Architecture: one bounded structured model call selects JSON parameters. Trusted Python
samples graphs and computes physics parameters; trusted React SVG renders static plots
and requestAnimationFrame simulations. Chat API multiplexes illustration and tutor events.
Manim remains available through its existing tool for explicit video workflows, not default chat.
Tech: Pydantic, Python ast/math, React SVG. No new runtime dependencies.
Spec: the JSON → template proposal and user instruction to implement all priorities in chat.

- [x] Add failing tests for domains, invalid expressions, template selection and unsupported requests.
- [x] Implement template schema, safe expression interpreter and scene payload generation.
- [x] Ground requests in extracted problem, current message and bounded conversation; broaden intent recognition.
- [x] Multiplex illustration work independently of tutor stream, preserve late results in history and cleanup tasks.
- [x] Add fixed React SVG templates, controls, explicit failed/skipped status and preserve results across reload.
- [x] Test ordering, persistence, parameter correctness, timeout and errors; run backend suite and frontend build.
- [x] Reload Docker and smoke test API and browser delivery without live code generation.

Constraints: keep five-second Manim timeout. Never execute model expressions via eval/sympify.
Unsupported topics must show a limit, not fabricate an unrelated diagram. Parameters/examples
must be visibly identified. Preserve existing video playback and injected graph service tests.
Review focus: singularity gaps, late result after text, cancellation, changed problem context,
invalid model parameters, browser animation cleanup/reduced motion, historical payload compatibility.

Validation: full backend suite, frontend production build, Edge/Playwright four-template UI checks, real model pendulum parameter smoke. Review issues (periodic graph sampling and new-problem context) fixed with regressions. Backend bind-mounted code reloaded, health200.
