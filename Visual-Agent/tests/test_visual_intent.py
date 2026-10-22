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
def test_declined_or_calculation_only_geometry_does_not_request_a_picture(question):
    assert not decide_visualization(question,'math').should_visualize
