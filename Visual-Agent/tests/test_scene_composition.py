import math

import pytest
from pydantic import ValidationError

from tools.visualization.scene import SceneSpec, compile_scene


def make_scene(**kwargs):
    return SceneSpec.model_validate({"title": "Example", **kwargs})


def test_new_projectile_scene_is_composed_without_a_topic_template():
    data = compile_scene(make_scene(duration=2, parameters=[{"name": "v", "value": 5}], objects=[
        {"id": "ball", "kind": "point", "points": [["v*t", "v*t-9.81*t^2/2"]], "label": "Vật"},
        {"id": "velocity", "kind": "arrow", "points": [["ball_x", "ball_y"], ["ball_x+v", "ball_y+v-9.81*t"]]},
    ]))
    assert data["template"] == "scene"
    assert data["objects"][0]["points"][0] == [0, 0, 0]
    last = data["frames"][-1]["objects"][0]["points"][0]
    assert last == pytest.approx([10, 10-9.81*2, 0])


def test_midpoint_and_intersection_relations_are_calculated():
    data = compile_scene(make_scene(objects=[
        {"id": "a", "kind": "point", "points": [[0, 0]]},
        {"id": "b", "kind": "point", "points": [[4, 2]]},
        {"id": "mid", "kind": "point"},
        {"id": "l1", "kind": "line", "points": [[0, 0], [2, 2]]},
        {"id": "l2", "kind": "line", "points": [[0, 2], [2, 0]]},
        {"id": "cross", "kind": "point"},
    ], relations=[
        {"target": "mid", "kind": "midpoint", "sources": ["a", "b"]},
        {"target": "cross", "kind": "intersection", "sources": ["l1", "l2"]},
    ]))
    objects = {item["id"]: item for item in data["objects"]}
    assert objects["mid"]["points"][0] == [2, 1, 0]
    assert objects["cross"]["points"][0] == pytest.approx([1, 1, 0])


@pytest.mark.parametrize("expression", ["__import__('os')", "a.x", "open('x')", "[1]", "missing+1", "x^9999"])
def test_scene_expressions_cannot_execute_code_or_unknown_names(expression):
    with pytest.raises(ValueError):
        compile_scene(make_scene(objects=[{"id": "p", "kind": "point", "points": [[expression, 0]]}]))


def test_cyclic_geometry_is_rejected():
    with pytest.raises(ValueError, match="reference|cycle"):
        compile_scene(make_scene(objects=[
            {"id": "a", "kind": "point", "points": [["b_x", 0]]},
            {"id": "b", "kind": "point", "points": [["a_x", 0]]},
        ]))


def test_3d_molecule_can_be_composed_from_spheres_and_bonds():
    data = compile_scene(make_scene(dimension=3, objects=[
        {"id": "o", "kind": "sphere", "points": [[0, 0, 0]], "radius": 0.4, "color": "#ef4444"},
        {"id": "h", "kind": "sphere", "points": [[1, 1, 0]], "radius": 0.2},
        {"id": "bond", "kind": "cylinder", "points": [[0, 0, 0], [1, 1, 0]], "radius": 0.08},
    ]))
    assert data["dimension"] == 3 and len(data["objects"]) == 3


def test_invalid_geometry_and_unbounded_scene_are_rejected():
    with pytest.raises(ValueError):
        compile_scene(make_scene(objects=[{"id": "c", "kind": "circle", "points": [[0, 0]], "radius": -1}]))
    with pytest.raises(ValidationError):
        make_scene(objects=[{"id": "x", "kind": "image", "url": "https://example.com"}])


def test_curve_supports_parameterized_two_dimensions():
    data = compile_scene(make_scene(objects=[{"id": "orbit", "kind": "curve", "x": "2*cos(u)", "y": "sin(u)", "u_min": 0, "u_max": 2*math.pi}]))
    curve = data["objects"][0]
    assert curve["segments"][0][0] == pytest.approx([2, 0, 0])


def test_time_sampling_cannot_alias_rapid_motion_into_a_static_point():
    with pytest.raises(ValueError, match="resolution"):
        compile_scene(make_scene(duration=10, objects=[{"id":"p","kind":"point","points":[["sin(1000*t)",0]]}]))


def test_nonplanar_3d_polygon_is_not_silently_flattened():
    with pytest.raises(ValueError, match="planar"):
        compile_scene(make_scene(dimension=3, objects=[{"id":"face","kind":"polygon","fill":True,"points":[[0,0,0],[1,0,0],[1,1,1],[0,1,0]]}]))


def test_relation_sources_must_have_the_right_geometry():
    with pytest.raises(ValueError, match="source"):
        compile_scene(make_scene(objects=[
            {"id":"axis","kind":"axes"},{"id":"p","kind":"point","points":[[1,0]]},
            {"id":"mid","kind":"point"}],
            relations=[{"target":"mid","kind":"midpoint","sources":["axis","p"]}]))


def test_bare_object_coordinate_uses_its_axis_and_tracks_animation():
    scene=make_scene(dimension=3,duration=1,objects=[
        {'id':'follower','kind':'point','points':[['anchor','anchor','anchor']]},
        {'id':'anchor','kind':'point','points':[['2+t',3,4]]},
    ])
    result=compile_scene(scene)
    for frame in result['frames']:
        objects={obj['id']:obj for obj in frame['objects']}
        assert objects['follower']['points']==objects['anchor']['points']
    assert result['objects'][0]['points'][0]==[2,3,4]


def test_bare_coordinate_does_not_override_a_parameter_with_the_same_name():
    scene=make_scene(parameters=[{'name':'anchor','value':7}],objects=[
        {'id':'anchor','kind':'point','points':[[2,3]]},
        {'id':'p','kind':'point','points':[['anchor',0]]},
    ])
    assert compile_scene(scene)['objects'][1]['points'][0][0]==7


def test_bare_coordinate_cycles_and_unknown_scalar_references_still_fail():
    with pytest.raises(ValueError,match='cycle'):
        compile_scene(make_scene(objects=[{'id':'a','kind':'point','points':[['a',0]]}]))
    with pytest.raises(ValueError,match='Unknown reference'):
        compile_scene(make_scene(objects=[
            {'id':'anchor','kind':'point','points':[[2,3]]},
            {'id':'c','kind':'circle','points':[[0,0]],'radius':'anchor'},
        ]))


def test_velocity_vector_can_be_zero_at_a_turning_point():
    result=compile_scene(make_scene(duration=1,objects=[
        {'id':'ball','kind':'point','points':[['2*cos(2*t)',0]]},
        {'id':'velocity','kind':'arrow','points':[['ball_x',0],['ball_x-4*sin(2*t)',0]]},
    ]))
    velocity=result['objects'][1]
    assert velocity['points'][0]==velocity['points'][1]
    assert any(frame['objects'][1]['points'][0]!=frame['objects'][1]['points'][1] for frame in result['frames'])


def test_zero_length_infinite_line_still_has_no_direction():
    with pytest.raises(ValueError,match='Degenerate'):
        compile_scene(make_scene(objects=[{'id':'line','kind':'line','points':[[0,0],[0,0]]}]))


def test_parallel_relation_requires_a_nonzero_source_direction():
    with pytest.raises(ValueError,match='Degenerate'):
        compile_scene(make_scene(objects=[
            {'id':'zero','kind':'arrow','points':[[0,0],[0,0]]},
            {'id':'p','kind':'point','points':[[1,1]]},
            {'id':'target','kind':'arrow'},
        ],relations=[{'target':'target','kind':'parallel','sources':['p','zero']}]))
