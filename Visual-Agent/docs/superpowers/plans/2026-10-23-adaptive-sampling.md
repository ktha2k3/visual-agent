# Adaptive scene sampling

Approved: replace topic-specific resolution choices with generic adaptive sampling.

Completed:
- Bound expression values and first derivatives using interval arithmetic.
- Refine space/time from chord error, derivative bounds and trigonometric phase;
  preserve formulas and share curve parameter knots across animation frames.
- Bound work, vertices, object/frame instances and the shared request deadline;
  report sampling failures without asking the model to rewrite the formula.
- Validate schema defaults, count failed repair attempts and save private
  diagnostic records identified in ordinary logs without problem text.
- Reviewer regression: treat atan2 branch cuts as discontinuities, never certify
  them with the local derivative formula.

Evidence: 156 backend tests, 2 frontend numeric tests, headless Edge checks for
2D/3D scenes, animation controls, exports and mobile. Live harmonic-motion request
completed in one attempt, 14.99 seconds including model generation, using 225
adaptive frames and 1,360 vertices. The previously failed original scene was not
saved, so this is a representative request, not a replay of its exact payload.
