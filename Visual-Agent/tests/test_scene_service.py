import asyncio

import pytest
import json

from tools.visualization.scene import SceneSpec
from tools.visualization.scene_service import SceneIllustrationService


@pytest.mark.asyncio
async def test_model_can_compose_an_unlisted_diagram():
    class Provider:
        async def structured(self, schema, *, system, prompt):
            assert schema is SceneSpec
            assert "thấu kính" in prompt
            return SceneSpec.model_validate({"title": "Thấu kính", "objects": [
                {"id": "lens", "kind": "ellipse", "points": [[0,0]], "size": [0.2,2,1]},
                {"id": "ray", "kind": "polyline", "points": [[-4,1],[0,1],[3,0]], "color": "#ef4444"},
                {"id": "axis", "kind": "line", "points": [[-5,0],[5,0]]},
            ]})
    result = await SceneIllustrationService(provider=Provider()).visualize(
        question="Vẽ tia sáng đi qua thấu kính", subject="physics", user_id="a")
    assert result.success and result.scene_data["template"] == "scene"
    assert len(result.scene_data["objects"]) == 3
    assert result.video_url is None


@pytest.mark.asyncio
async def test_missing_evidence_and_invalid_geometry_do_not_become_success():
    class Provider:
        async def structured(self, *args, **kwargs):
            return SceneSpec(reason="Cần ảnh giải phẫu có nguồn")
    result = await SceneIllustrationService(provider=Provider()).visualize(question="giải phẫu", subject="physics", user_id="a")
    assert not result.success and result.status == "unsupported"


@pytest.mark.asyncio
async def test_scene_model_deadline_is_bounded():
    class Provider:
        async def structured(self, *args, **kwargs):
            await asyncio.sleep(100)
    result = await SceneIllustrationService(provider=Provider(), timeout=0.01).visualize(
        question="hình", subject="math", user_id="a")
    assert not result.success and result.status == "failed"


@pytest.mark.asyncio
async def test_one_bounded_data_repair_receives_the_validation_error():
    class Provider:
        calls = 0
        async def structured(self, *args, **kwargs):
            raise ValueError("Unknown reference: bad_x")
        async def text(self, *, system, prompt):
            self.calls += 1
            assert "bad_x" in prompt
            return '{"title":"A","objects":[{"id":"a","kind":"point","points":[[0,0]]}]}'
    provider = Provider()
    result = await SceneIllustrationService(provider=provider).visualize(question="Vẽ điểm A",subject="math",user_id="a")
    assert result.success and provider.calls == 1


@pytest.mark.asyncio
async def test_failed_second_attempt_is_counted_and_both_invalid_scenes_are_saved(tmp_path):
    class Provider:
        async def structured(self,*args,**kwargs):
            return SceneSpec.model_validate({"objects":[{"id":"bad","kind":"point","radius":-1,"points":[[0,0]]}]})
        async def text(self,**kwargs):
            return '{"objects":[{"id":"still_bad","kind":"point","radius":-2,"points":[[0,0]]}]}'
    result=await SceneIllustrationService(provider=Provider(),diagnostic_dir=tmp_path).visualize(
        question="draw",subject="math",user_id="owner")
    assert not result.success and result.attempts==2
    records=[json.loads(path.read_text()) for path in sorted(tmp_path.glob('*.json'))]
    assert [record['attempt'] for record in records]==[1,2]
    assert records[1]['scene']['objects'][0]['id']=='still_bad'
    assert records[1]['error_type']=='ValueError'
    assert 'user_id' not in records[1] and records[1]['owner_hash']
    assert result.diagnostic_id


@pytest.mark.asyncio
async def test_sampling_budget_failure_is_explicit_and_not_sent_for_model_repair(tmp_path):
    class Provider:
        async def structured(self,*args,**kwargs):
            return SceneSpec.model_validate({"objects":[{"id":"f","kind":"curve","x":"u","y":"sin(10000*u)"}]})
        async def text(self,**kwargs):
            pytest.fail('Resource-limit failures must not rewrite the requested formula')
    result=await SceneIllustrationService(provider=Provider(),diagnostic_dir=tmp_path).visualize(question="plot",subject="math",user_id="owner")
    assert not result.success and result.attempts==1
    assert 'thu hẹp' in result.error
    assert list(tmp_path.glob('*.json'))


@pytest.mark.asyncio
async def test_bare_coordinate_reference_needs_no_second_model_call(tmp_path):
    class Provider:
        async def structured(self,*args,**kwargs):
            return SceneSpec.model_validate({'duration':1,'objects':[
                {'id':'block','kind':'box','points':[['x0',0]],'size':[0.5,0.5,1]},
                {'id':'x0','kind':'point','points':[['2*cos(t)',0]]},
            ]})
        async def text(self,**kwargs):
            pytest.fail('A coordinate shorthand must not require a model repair')
    result=await SceneIllustrationService(provider=Provider(),diagnostic_dir=tmp_path).visualize(
        question='illustrate motion',subject='physics',user_id='owner')
    assert result.success and result.attempts==1
