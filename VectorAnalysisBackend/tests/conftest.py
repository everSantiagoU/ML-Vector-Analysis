import pytest


@pytest.fixture(params=[
    '<t,t>', '<t,t,t,t>', '<t,,t>', '<,t,t>', '<t,t,>',
    '[t,t,t]', '<t,t,t)', '(t,t,t>', '<(t,t,t>', '<t),t,t>',
    '<t,x,t>', 'prefix=<t,t,t>', 'prefix r(t)=<t,t,t>',
    'r(x)=<t,t,t>', 'r(t,x)=<t,t,t>', 'r(t)=s(t)=<t,t,t>',
    '= <t,t,t>', 'r(t)==<t,t,t>', 'r()=<t,t,t>',
    '<[t],t,t>', '<(t,t),t,t>', '<{t},t,t>', '<sin,t,t>',
    '<t.real,t,t>', '<t[0],t,t>', '<sin(t,1),t,t>', '<log(),t,t>',
    '<log(t,2,3),t,t>', '<sin(t,x=1),t,t>', '<t**,t,t>',
    None, 1, [], '',
])
def invalid_vector(request):
    return request.param


@pytest.fixture(params=[
    't.real', 't[0]', "'t'", '[t]', '(t,t)', '{t}', "{'a':t}",
    '[t for t in (1,2)]', '(t for t in (1,2))', 'lambda t:t',
    'sin(t, evaluate=False)', 'sin(*[t])', "log(t, **{'base':2})",
    'unknown(t)', '__import__("math")', 'True', 'None', '1j',
    't < 2', 't and t', 't if t else 0', '(x:=t)', 't//2', 't%2',
    'sin(t,1)', 'sqrt(t,2)', 'exp()', 'log(t,2,3)', 'sin',
    'x+1', '2t', '', None, 1, [],
])
def invalid_component(request):
    return request.param
