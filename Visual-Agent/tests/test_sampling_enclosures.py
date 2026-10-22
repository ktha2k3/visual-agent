import math

import pytest

from tools.visualization.scene import Expression
from tools.visualization.sampling import Dual


def test_atan2_branch_cut_has_no_finite_derivative_certificate():
    expression=Expression('atan2(u,-1)',{'u'})
    result=expression.enclosure({'u':Dual((-0.1,0.1),(1,1))})
    assert result.derivative is None or not all(math.isfinite(v) for v in result.derivative)


@pytest.mark.parametrize('source,interval',[
    ('sin(u)*cos(u)',(-2,3)), ('1/(1+u^2)',(-2,2)), ('exp(-u^2)',(-2,2)),
    ('sqrt(u)',(0.1,2)), ('log(u)',(0.2,2)), ('abs(u)',(-1,1)),
    ('asin(u)',(-0.8,0.7)),('acos(u)',(-0.8,0.7)),('atan(u)',(-2,2)),
    ('atan2(u,2)',(-2,2)),('u^3-u',(-2,2)),('u^(-2)',(0.5,2)),
])
def test_interval_value_and_derivative_enclose_numeric_samples(source,interval):
    expression=Expression(source,{'u'})
    result=expression.enclosure({'u':Dual(interval,(1,1))})
    for i in range(1,100):
        x=interval[0]+(interval[1]-interval[0])*i/100
        value=expression({'u':x})
        assert result.value[0]-1e-8<=value<=result.value[1]+1e-8
        h=1e-6
        derivative=(expression({'u':x+h})-expression({'u':x-h}))/(2*h)
        assert result.derivative[0]-1e-5<=derivative<=result.derivative[1]+1e-5
