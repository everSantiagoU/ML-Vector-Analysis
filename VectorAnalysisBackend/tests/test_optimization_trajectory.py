import json

import numpy as np
import pytest

from mlvectoranalysis.domain.multivariable_function import MultivariableFunction
from mlvectoranalysis.optimization import GradientDescent, GradientDescentConfig, TrajectoryRecorder
from mlvectoranalysis.visualization import OptimizationTrajectory


@pytest.mark.parametrize('mode', ['sampled', 'metrics', 'none'])
@pytest.mark.parametrize('max_points', [2, 3, 7, 100])
def test_history_is_strictly_bounded_deterministic_owned(mode, max_points):
    recorders = [TrajectoryRecorder(mode, max_points, (0, 2)) for _ in range(2)]
    p = np.zeros(4)
    for iteration in range(2501):
        p[:] = iteration
        for recorder in recorders:
            recorder.record(iteration, p, float(iteration), 1., None if iteration == 0 else .1)
            assert len(recorder._points) <= max_points
    p[:] = -1
    a, b = recorders
    assert [s.to_dict() for s in a.snapshots] == [s.to_dict() for s in b.snapshots]
    if mode == 'none':
        assert a.snapshots == ()
    else:
        indices = [s.iteration for s in a.snapshots]
        assert indices == sorted(set(indices))
        assert indices[0] == 0 and indices[-1] == 2500
        if mode == 'sampled':
            assert sum(s.parameters.nbytes for s in a.snapshots) <= max_points*2*8
            for s in a.snapshots:
                np.testing.assert_array_equal(s.parameters, [s.iteration]*2)
        else:
            assert all(s.parameters is None for s in a.snapshots)


def test_sampled_engine_long_run_and_json_projection():
    class Linear:
        dimension = 8
        def value(self, p): return float(p.sum())
        def value_and_grad(self, p): return self.value(p), np.ones(8)
    r = GradientDescent(GradientDescentConfig(max_iter=1200, alpha=.1, step_strategy='fixed',
        history_indices=(2, 5), max_points=11)).minimize(Linear(), np.zeros(8))
    assert r.n_iter == 1200 and len(r.snapshots) <= 11
    assert r.parameters.shape == (8,)
    assert r.snapshots[-1].iteration == 1200
    assert r.snapshots[0].step is None
    with pytest.raises(ValueError, match='Select'):
        OptimizationTrajectory(r)
    trajectory = OptimizationTrajectory(r, axes=(5, 2))
    payload = trajectory.to_dict()
    assert payload['is_projection'] and payload['sampled']
    assert payload['original_indices'] == [5, 2]
    assert payload['axes'] == ['parameter_5', 'parameter_2']
    json.dumps(payload, allow_nan=False)
    json.dumps(r.to_dict(), allow_nan=False)
    for s in r.snapshots:
        np.testing.assert_allclose(trajectory.interpolate(s.iteration), s.parameters[::-1])
    left, right = r.snapshots[:2]
    np.testing.assert_allclose(trajectory.interpolate((left.iteration+right.iteration)/2),
                               (left.parameters[::-1]+right.parameters[::-1])/2)
    with pytest.raises(ValueError):
        OptimizationTrajectory(r, axes=(0, 2))
    for invalid in (-1, 1201, np.nan, True, '1'):
        with pytest.raises(ValueError):
            trajectory.interpolate(invalid)
    r.snapshots[0].parameters[:] = 100
    assert not np.any(r.parameters == 100)
    assert not np.any(r.snapshots[-1].parameters == 100)


def test_single_point_and_metrics_payload():
    f = MultivariableFunction('x^2+y^2+z^2')
    r = GradientDescent().minimize(f, [0, 0, 0])
    trajectory = OptimizationTrajectory(r)
    np.testing.assert_array_equal(trajectory.interpolate(0), [0, 0, 0])
    assert trajectory.to_dict()['coordinate_system'] == 'xyz'
    assert not trajectory.to_dict()['is_projection']
    with pytest.raises(ValueError):
        trajectory.interpolate(.01)
    for mode in ('metrics', 'none'):
        r = GradientDescent(GradientDescentConfig(history_mode=mode)).minimize(f, [1, 1, 1])
        t = OptimizationTrajectory(r)
        assert t.to_dict()['points'] == []
        assert len(t.to_dict()['losses']) == len(r.snapshots)
        with pytest.raises(ValueError):
            t.interpolate(0)


def test_one_parameter_can_export_loss_without_fabricated_axes():
    from mlvectoranalysis.optimization import LinearLeastSquaresObjective
    result = GradientDescent().minimize(LinearLeastSquaresObjective([[1], [2]], [2, 4], fit_intercept=False), [0])
    payload = OptimizationTrajectory(result).to_dict()
    assert payload['axes'] == payload['points'] == []
    assert payload['losses'] == [5, 0]
