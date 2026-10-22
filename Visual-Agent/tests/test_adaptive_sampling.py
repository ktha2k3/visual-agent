import math

import pytest

from tools.visualization.scene import SceneSpec, compile_scene


def curve_scene(expression, *, samples=16, domain=(-5,5), viewport=(-5,-2,5,2)):
    return compile_scene(SceneSpec.model_validate({"viewport":viewport,"objects":[{
        "id":"f","kind":"curve","x":"u","y":expression,"samples":samples,
        "u_min":domain[0],"u_max":domain[1],
    }]}))


def test_compiler_increases_resolution_without_changing_requested_function():
    result=curve_scene("cos(4*pi*u)")
    points=[point for segment in result["objects"][0]["segments"] for point in segment]
    assert len(points)>160
    assert all(y==pytest.approx(math.cos(4*math.pi*x)) for x,y,z in points)
    assert max(point[1] for point in points)>0.99
    assert min(point[1] for point in points)<-0.99


def test_geometric_error_refines_a_nontrigonometric_local_feature():
    result=curve_scene("exp(-100*(u-0.123)^2)", domain=(-1,1), viewport=(-1,-0.1,1,1.1))
    points=result["objects"][0]["segments"][0]
    near=sum(abs(point[0]-0.123)<0.3 for point in points)
    far=sum(abs(point[0]-0.123)>0.7 for point in points)
    assert near>far
    assert max(point[1] for point in points)>0.98


def test_midpoint_aliasing_does_not_make_periodic_graph_flat():
    result=curve_scene("sin(32*pi*u)",domain=(0,1),viewport=(0,-1.2,1,1.2))
    points=[p for segment in result["objects"][0]["segments"] for p in segment]
    assert max(p[1] for p in points)>0.9 and min(p[1] for p in points)<-0.9


def test_poles_are_gaps_not_vertical_connections():
    result=curve_scene("1/u",domain=(-2,2),viewport=(-2,-4,2,4))
    segments=result["objects"][0]["segments"]
    assert len(segments)>=2
    assert all(not (segment[0][0]<0<segment[-1][0]) for segment in segments)


def test_resource_limit_is_explicit_instead_of_underresolved_success():
    with pytest.raises(ValueError,match="resolution"):
        curve_scene("sin(10000*u)")


def test_time_sampling_increases_for_motion_within_budget():
    result=compile_scene(SceneSpec.model_validate({"duration":2,"objects":[{
        "id":"p","kind":"point","points":[["sin(6*pi*t)",0]]}]}))
    assert len(result["frames"])>30
    xs=[frame["objects"][0]["points"][0][0] for frame in result["frames"]]
    assert max(xs)>0.95 and min(xs)<-0.95


def test_default_values_serialize_without_pydantic_warnings():
    spec=SceneSpec.model_validate({"objects":[{"id":"p","kind":"point","points":[[0,0]]}]})
    spec.model_dump_json(warnings="error")


def test_off_grid_narrow_feature_is_not_silently_missed():
    result=curve_scene("exp(-10000*10000*(u-0.12345)^2)",domain=(-1,1),viewport=(-1,-0.1,1,1.1))
    points=[p for segment in result["objects"][0]["segments"] for p in segment]
    assert max(p[1] for p in points)>0.98


def test_off_grid_short_pulse_is_captured_in_animation():
    result=compile_scene(SceneSpec.model_validate({"duration":1,"objects":[{
        "id":"p","kind":"point","points":[["t","exp(-10000*10000*(t-0.12345)^2)"]]}]}))
    assert max(frame["objects"][0]["points"][0][1] for frame in result["frames"])>0.98


def test_animated_curves_share_parameter_knots_across_frames():
    result=compile_scene(SceneSpec.model_validate({"duration":0.5,"objects":[{
        "id":"wave","kind":"curve","x":"u","y":"sin(u-t)","u_min":-2,"u_max":2}]}))
    curves=[frame["objects"][0]["segments"][0] for frame in result["frames"]]
    assert all([p[0] for p in curve]==[p[0] for p in curves[0]] for curve in curves)
    assert result['sampling']['vertices']<=60000
    assert all('_knots' not in frame['objects'][0] for frame in result['frames'])


def test_compiler_respects_work_deadline_even_in_worker_thread():
    import time
    spec=SceneSpec.model_validate({'objects':[{'id':'f','kind':'curve','x':'u','y':'sin(u)'}]})
    with pytest.raises(ValueError,match='resolution'):
        compile_scene(spec,deadline=time.monotonic()-1)


def test_narrow_atan2_branch_pulse_is_captured_or_explicitly_rejected():
    from tools.visualization.sampling import SamplingLimitError
    try:
        result=curve_scene('atan2(u-0.12345,-1)-atan2(u-0.12346,-1)',domain=(-1,1),viewport=(-1,-7,1,7))
    except SamplingLimitError:
        return
    points=[p for segment in result['objects'][0]['segments'] for p in segment]
    assert max(abs(p[1]) for p in points)>6
