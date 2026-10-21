import numpy as np
import pytest

from mlvectoranalysis.ml import LinearRegressionGD
from mlvectoranalysis.optimization import GradientDescentConfig


TIGHT = GradientDescentConfig(max_iter=4000, gradient_atol=1e-10, gradient_rtol=1e-10)


@pytest.mark.parametrize('standardize', [False, True])
@pytest.mark.parametrize('fit_intercept', [False, True])
def test_exact_recovery_in_original_units(standardize, fit_intercept):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(160, 4))*[1, 2, 3, 4]+[.5, 1, -2, 0]
    coef = np.array([.2, -1, 3, 2])
    intercept = 4 if fit_intercept else 0
    targets = X@coef+intercept
    model = LinearRegressionGD(standardize=standardize, fit_intercept=fit_intercept, config=TIGHT)
    assert model.fit(X, targets) is model
    np.testing.assert_allclose(model.coef_, coef, atol=3e-8)
    assert model.intercept_ == pytest.approx(intercept, abs=3e-8)
    np.testing.assert_allclose(model.predict(X), targets, atol=2e-7)
    assert model.optimization_result_.converged
    assert model.optimization_result_.coordinate_system == ('standardized_parameters' if standardize else 'original_parameters')
    # Inference uses original-unit coefficients; no dependence on training X.
    X_new = rng.normal(size=(20, 4))
    expected = X_new@coef+intercept
    X[:] = 1e6
    np.testing.assert_allclose(model.predict(X_new), expected, atol=2e-7)


@pytest.mark.parametrize('case', ['noisy', 'rank_deficient', 'constant_column', 'zero_X',
                                 'constant_targets', 'one_row', 'one_feature', 'wide'])
@pytest.mark.parametrize('fit_intercept', [False, True])
def test_predictions_and_loss_match_lstsq(case, fit_intercept):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 4))
    if case == 'rank_deficient':
        X[:, 2] = X[:, 0]+2*X[:, 1]
    elif case == 'constant_column':
        X[:, 2] = 5
    elif case == 'zero_X':
        X[:] = 0
    elif case == 'one_row':
        X = X[:1]
    elif case == 'one_feature':
        X = X[:, :1]
    elif case == 'wide':
        X = X[:2]
    targets = X@rng.normal(size=X.shape[1])+2+rng.normal(size=len(X))*.05
    if case == 'constant_targets':
        targets[:] = 4
    A = np.column_stack([X, np.ones(len(X))]) if fit_intercept else X
    oracle, *_ = np.linalg.lstsq(A, targets, rcond=None)
    model = LinearRegressionGD(fit_intercept=fit_intercept, standardize=True, config=TIGHT).fit(X, targets)
    predictions = model.predict(X)
    np.testing.assert_allclose(predictions, A@oracle, atol=2e-7, rtol=2e-7)
    expected_loss = np.mean((A@oracle-targets)**2)/2
    assert model.optimization_result_.loss == pytest.approx(expected_loss, abs=1e-12)


def test_predict_validation_and_fit_replacement():
    model = LinearRegressionGD(standardize=True)
    with pytest.raises(ValueError, match='fit'):
        model.predict([[1]])
    model.fit([[1], [2]], [2, 4])
    with pytest.raises(ValueError):
        model.predict([[1, 2]])
    with pytest.raises(ValueError):
        model.predict([[1j]])
    first = model.optimization_result_
    model.fit([[1, 1], [2, 1], [3, 1]], [2, 3, 4])
    assert model.optimization_result_ is not first
    assert model.coef_.shape == (2,)
    assert model.mean_.shape == (2,)
    with pytest.raises(ValueError):
        model.fit([[1]], [[1]])
    assert model.coef_ is model.intercept_ is model.optimization_result_ is None
    assert model.mean_ is model.scale_ is model.n_features_in_ is None
    with pytest.raises(ValueError):
        model.predict([[1]])
