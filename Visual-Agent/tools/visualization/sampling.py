"""Geometry-driven subdivision and conservative interval/derivative enclosures."""
from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass

UNKNOWN = (-math.inf, math.inf)
ZERO = (0.0, 0.0)


class SamplingLimitError(ValueError):
    pass


@dataclass
class WorkBudget:
    remaining: int = 120000
    deadline: float | None = None

    def spend(self, count=1):
        self.remaining -= count
        if self.remaining < 0 or (self.deadline is not None and time.monotonic() >= self.deadline):
            raise SamplingLimitError("Sampling resolution exceeds the scene work budget")


def add(a, b):
    return None if a is None or b is None else (a[0]+b[0], a[1]+b[1])


def neg(a):
    return None if a is None else (-a[1], -a[0])


def mul(a, b):
    if a is None or b is None:
        return None
    if a == ZERO or b == ZERO:
        return ZERO
    values = [x*y for x in a for y in b]
    return UNKNOWN if any(math.isnan(v) for v in values) else (min(values),max(values))


def div(a, b):
    if a is None or b is None:
        return None
    if b[0] <= 0 <= b[1]:
        return UNKNOWN
    return mul(a, (1/b[1],1/b[0]))


def power(a, exponent):
    if a is None:
        return None
    if exponent == 0:
        return (1.0,1.0)
    if exponent < 0 and a[0] <= 0 <= a[1]:
        return UNKNOWN
    if not float(exponent).is_integer() and a[0] < 0:
        return None if a[1] < 0 else UNKNOWN
    try:
        values = [a[0]**exponent,a[1]**exponent]
        if exponent > 0 and a[0] <= 0 <= a[1]:
            values.append(0.0)
        if any(isinstance(v,complex) or math.isnan(v) for v in values):
            return UNKNOWN
        return min(values),max(values)
    except (OverflowError,ValueError,ZeroDivisionError):
        return UNKNOWN


def trig(a, name):
    if a is None:
        return None
    lo,hi = a
    if not math.isfinite(lo) or not math.isfinite(hi):
        return UNKNOWN if name == "tan" else (-1.0,1.0)
    if name == "tan":
        if math.ceil((lo-math.pi/2)/math.pi) <= math.floor((hi-math.pi/2)/math.pi):
            return UNKNOWN
        return math.tan(lo),math.tan(hi)
    if hi-lo >= 2*math.pi:
        return (-1.0,1.0)
    function = math.sin if name == "sin" else math.cos
    values = [function(lo),function(hi)]
    shift = math.pi/2 if name == "sin" else 0
    for index in range(math.ceil((lo-shift)/math.pi),math.floor((hi-shift)/math.pi)+1):
        values.append(1.0 if index%2 == 0 else -1.0)
    return min(values),max(values)


@dataclass(frozen=True)
class Dual:
    value: tuple[float,float] | None
    derivative: tuple[float,float] | None = ZERO

    @classmethod
    def constant(cls, value):
        return cls((float(value),float(value)))

    def __add__(self, other):
        other=other if isinstance(other,Dual) else Dual.constant(other)
        return Dual(add(self.value,other.value),add(self.derivative,other.derivative))

    def __neg__(self):
        return Dual(neg(self.value),neg(self.derivative))

    def __sub__(self, other):
        return self + -(other if isinstance(other,Dual) else Dual.constant(other))

    def __mul__(self, other):
        other=other if isinstance(other,Dual) else Dual.constant(other)
        return Dual(mul(self.value,other.value),add(mul(self.derivative,other.value),mul(self.value,other.derivative)))

    def __truediv__(self, other):
        other=other if isinstance(other,Dual) else Dual.constant(other)
        return Dual(div(self.value,other.value),div(add(mul(self.derivative,other.value),neg(mul(self.value,other.derivative))),power(other.value,2)))


def enclosure(node, environment):
    """Interval arithmetic + chain rule; no user expressions are executed."""
    if isinstance(node, ast.Expression):
        return enclosure(node.body,environment)
    if isinstance(node,ast.Constant):
        return Dual.constant(node.value)
    if isinstance(node,ast.Name):
        if node.id in {"pi","e"}:
            return Dual.constant(math.pi if node.id == "pi" else math.e)
        return environment[node.id]
    if isinstance(node,ast.UnaryOp):
        value=enclosure(node.operand,environment)
        return Dual(neg(value.value),neg(value.derivative)) if isinstance(node.op,ast.USub) else value
    if isinstance(node,ast.BinOp):
        a,b=enclosure(node.left,environment),enclosure(node.right,environment)
        if isinstance(node.op,ast.Add):
            return Dual(add(a.value,b.value),add(a.derivative,b.derivative))
        if isinstance(node.op,ast.Sub):
            return Dual(add(a.value,neg(b.value)),add(a.derivative,neg(b.derivative)))
        if isinstance(node.op,ast.Mult):
            return Dual(mul(a.value,b.value),add(mul(a.derivative,b.value),mul(a.value,b.derivative)))
        if isinstance(node.op,ast.Div):
            return Dual(div(a.value,b.value),div(add(mul(a.derivative,b.value),neg(mul(a.value,b.derivative))),power(b.value,2)))
        exponent=b.value[0]
        return Dual(power(a.value,exponent),mul(mul((exponent,exponent),power(a.value,exponent-1)),a.derivative))
    values=[enclosure(arg,environment) for arg in node.args]
    if any(value.value is None for value in values):
        return Dual(None,None)
    name=node.func.id
    a=values[0];lo,hi=a.value
    if name in {"sin","cos","tan"}:
        derivative = trig(a.value,"cos") if name == "sin" else neg(trig(a.value,"sin")) if name == "cos" else div((1,1),power(trig(a.value,"cos"),2))
        return Dual(trig(a.value,name),mul(derivative,a.derivative))
    if name == "abs":
        value=(0.0,max(abs(lo),abs(hi))) if lo <= 0 <= hi else (min(abs(lo),abs(hi)),max(abs(lo),abs(hi)))
        sign=(-1,1) if lo < 0 < hi else (-1,-1) if hi <= 0 else (1,1)
        return Dual(value,mul(sign,a.derivative))
    if name in {"min","max"}:
        b=values[1];function=min if name == "min" else max
        value=(function(lo,b.value[0]),function(hi,b.value[1]))
        derivatives=[a.derivative,b.derivative]
        derivative=UNKNOWN if any(d is None for d in derivatives) else (min(d[0] for d in derivatives),max(d[1] for d in derivatives))
        return Dual(value,derivative)
    if name == "atan2":
        b=values[1]
        # atan2(y,x) jumps by 2*pi across y=0 when x<0. The local
        # derivative formula cannot certify an interval crossing that cut.
        if b.value[0]<=0 and a.value[0]<=0<=a.value[1]:
            return Dual((-math.pi,math.pi),UNKNOWN)
        return Dual((-math.pi,math.pi),div(add(mul(b.value,a.derivative),neg(mul(a.value,b.derivative))),add(power(a.value,2),power(b.value,2))))
    try:
        if name == "sqrt":
            if hi < 0:
                return Dual(None,None)
            if lo < 0:
                return Dual(UNKNOWN,UNKNOWN)
            value=(math.sqrt(lo),math.sqrt(hi));derivative=div((1,1),mul((2,2),value))
        elif name == "log":
            if hi <= 0:
                return Dual(None,None)
            if lo <= 0:
                return Dual(UNKNOWN,UNKNOWN)
            value=(math.log(lo),math.log(hi));derivative=div((1,1),a.value)
        elif name == "exp":
            value=(math.exp(lo),math.exp(hi));derivative=value
        elif name == "atan":
            value=(math.atan(lo),math.atan(hi));derivative=div((1,1),add((1,1),power(a.value,2)))
        else:  # asin/acos
            if hi < -1 or lo > 1:
                return Dual(None,None)
            if lo < -1 or hi > 1:
                return Dual(UNKNOWN,UNKNOWN)
            value=(math.asin(lo),math.asin(hi)) if name == "asin" else (math.acos(hi),math.acos(lo))
            squared=add((1,1),neg(power(a.value,2)))
            root=None if squared is None else (math.sqrt(max(0,squared[0])),math.sqrt(max(0,squared[1])))
            derivative=div((1,1),root)
            if name == "acos":derivative=neg(derivative)
        return Dual(value,mul(derivative,a.derivative))
    except (ArithmeticError,ValueError):
        return Dual(UNKNOWN,UNKNOWN)


def phase_variation(expressions, environment, span):
    result=0.0
    for expression in expressions:
        for node in ast.walk(expression.tree):
            if isinstance(node,ast.Call) and node.func.id in {"sin","cos","tan"}:
                derivative=enclosure(node.args[0],environment).derivative
                variation=math.inf if derivative is None else max(abs(v) for v in derivative)*span
                result=max(result,variation)
    return result


def adaptive_sample(evaluate, start, end, *, tolerance, budget, phase=None, domain=None,
                    envelope=None, error_bound=None, seed_count=8, max_points=4096, gaps=False, linear=False):
    """Return ordered (parameter, vector-or-None) samples within a hard budget.

    Interval phase bounds prevent periodic aliasing. Off-grid probes and chord
    errors refine local nontrigonometric geometry. Unresolved precision is an
    explicit failure, never a silently accepted coarse curve.
    """
    cache={}
    def get(value):
        if value not in cache:
            budget.spend()
            try:
                vector=tuple(evaluate(value))
                if not all(math.isfinite(v) for v in vector):raise ValueError("Nonfinite sample")
                cache[value]=vector
            except SamplingLimitError:
                raise
            except (ArithmeticError,ValueError):
                if not gaps:raise
                cache[value]=None
        return cache[value]

    output=[]
    def emit(value,point):
        if output and output[-1][0] == value:
            output[-1]=(value,point)
        else:output.append((value,point))
        if len(output)>max_points:
            raise SamplingLimitError("Required sampling resolution exceeds the point budget")

    def subdivide(a,b,depth):
        budget.spend()
        state=domain(a,b) if domain else "valid"
        if state == "outside":
            emit(a,None);emit(b,None);return
        left,right=get(a),get(b)
        fractions=(0.211324865405187,0.25,0.5,0.75,0.788675134594813)
        probes=[get(a+(b-a)*fraction) for fraction in fractions]
        invalid=left is None or right is None or any(point is None for point in probes)
        error=math.inf
        if not invalid:
            direction=[y-x for x,y in zip(left,right)]
            norm=sum(value*value for value in direction)
            errors=[]
            for fraction,point in zip(fractions,probes):
                projection=fraction if linear else max(0,min(1,sum((value-origin)*delta for value,origin,delta in zip(point,left,direction))/norm)) if norm else 0
                errors.append(math.sqrt(sum((value-origin-projection*delta)**2 for value,origin,delta in zip(point,left,direction))))
            error=max(errors)
        variation=phase(a,b) if phase else 0
        if error_bound:error=max(error,error_bound(a,b))
        enclosed=True
        if envelope and not invalid:
            bounds=envelope(a,b)
            cloud=[left,*probes,right]
            derivative_errors=[]
            for index,bound in enumerate(bounds):
                value=bound.value if isinstance(bound,Dual) else bound
                derivative=bound.derivative if isinstance(bound,Dual) else None
                if value is not None and all(math.isfinite(v) for v in value) and derivative is not None and all(math.isfinite(v) for v in derivative):
                    slope=(right[index]-left[index])/(b-a)
                    # The error is zero at both endpoints. A bound on the
                    # derivative of (curve - chord) bounds its interior error.
                    derivative_errors.append(max(abs(v-slope) for v in derivative)*(b-a)/2)
                elif value is None or value[0]<min(point[index] for point in cloud)-tolerance or value[1]>max(point[index] for point in cloud)+tolerance:
                    enclosed=False;break
            if math.sqrt(sum(value*value for value in derivative_errors))>tolerance:
                enclosed=False
        if not invalid and enclosed and error<=tolerance and variation<=math.pi/4*(1+1e-10):
            emit(a,left);emit(b,right);return
        if depth>=20 or b-a <= (end-start)*1e-10:
            if gaps and (invalid or state == "uncertain"):
                emit(a,left);emit((a+b)/2,None);emit(b,right);return
            raise SamplingLimitError("Cannot meet sampling resolution within the subdivision budget")
        middle=(a+b)/2
        subdivide(a,middle,depth+1);subdivide(middle,b,depth+1)

    for index in range(seed_count):
        subdivide(start+(end-start)*index/seed_count,start+(end-start)*(index+1)/seed_count,0)
    return output
