import pytest

from tools.visualization.routing import decide_visualization


@pytest.mark.parametrize('question',[
    'đưa ra hình mình hoạ cho đường thẳng tiếp tuyến mặt cầu',
    'Đưa ra hình cho đường thẳng tiếp tuyến mặt cầu',
    'Cung cấp một hình về hình chóp',
    'Cho em hình tiếp tuyến của mặt cầu',
    'Hiển thị đồ thị y=x^2',
])
def test_explicit_picture_request_routes_to_scene_generation(question):
    assert decide_visualization(question,'math').should_visualize


@pytest.mark.parametrize('question',[
    'Không cần hình, giải thích tiếp tuyến mặt cầu',
    'Đừng cung cấp hình, chỉ giải thích',
    'Tính diện tích hình tròn bán kính 2',
    'Cho hình chóp S.ABC, tính thể tích',
])
def test_every_supported_question_requests_a_picture(question):
    assert decide_visualization(question,'math').should_visualize


@pytest.mark.parametrize('subject',['math','physics'])
@pytest.mark.parametrize('question',['2+2','x=2','Em chưa hiểu','Chào bạn'])
def test_no_keywords_are_required_for_automatic_illustration(subject,question):
    decision=decide_visualization(question,subject)
    assert decision.should_visualize and decision.reason=='automatic_illustration'


@pytest.mark.asyncio
async def test_clarification_turn_still_starts_independent_illustration():
    from education.tutor_graph import build_tutor_graph
    from services.repository import MemoryRepository
    from tools.visualization.streaming import illustration_stream
    from tools.visualization.models import StemVisualizationResult

    calls=[]
    class Service:
        async def visualize(self,**kwargs):
            calls.append(kwargs)
            return StemVisualizationResult(subject='math',success=False,status='unsupported',
                                           error='Need problem data')
    graph=build_tutor_graph(MemoryRepository(),object(),research_service=object())
    events=[event async for event in illustration_stream(graph,{
        'user_id':'student','session_id':'auto-picture','subject':'math',
        'message':'Em chưa hiểu','attachment_ids':[], 'visualization_external':True,
    },{},service=Service())]
    assert len(calls)==1
    updates=[payload for mode,payload in events if mode=='updates']
    assert any('clarification' in update for update in updates)
    assert any('visualization_tool' in update for update in updates)
