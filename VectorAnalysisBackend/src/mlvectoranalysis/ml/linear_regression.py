"""Single-target batch regression; coefficients exposed in original units."""
import numpy as np

from ..optimization import GradientDescent, GradientDescentConfig, LinearLeastSquaresObjective
from ..optimization._validation import real_array


class LinearRegressionGD:
    def __init__(self, *, fit_intercept=True, standardize=False, block_size=None,
                 config: GradientDescentConfig | None = None):
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.block_size = block_size
        self.config = config if config is not None else GradientDescentConfig()
        self._clear()

    def _clear(self):
        self.coef_ = self.intercept_ = self.optimization_result_ = None
        self.mean_ = self.scale_ = self.n_features_in_ = None

    def fit(self, X, targets):
        # A failed fit leaves an unfitted model, never a mixture of two fits.
        self._clear()
        objective = LinearLeastSquaresObjective(X, targets, fit_intercept=self.fit_intercept,
                                               standardize=self.standardize, block_size=self.block_size)
        result = GradientDescent(self.config).minimize(objective, np.zeros(objective.dimension))
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            coef = result.parameters[:objective.d] / objective.scale
            intercept = float((result.parameters[-1] if self.fit_intercept else 0.0) - np.dot(objective.mean, coef))
        if not np.isfinite(coef).all() or not np.isfinite(intercept):
            raise ValueError('Original coefficients exceed float64 range.')
        self.coef_, self.intercept_, self.optimization_result_ = coef, intercept, result
        self.mean_, self.scale_ = objective.mean.copy(), objective.scale.copy()
        self.n_features_in_ = objective.d
        return self

    def predict(self, X_new):
        if self.coef_ is None:
            raise ValueError('Call fit before predict.')
        X_new = real_array(X_new, 'X_new', 2)
        if X_new.shape[1] != self.n_features_in_:
            raise ValueError(f'X_new must have {self.n_features_in_} features.')
        try:
            with np.errstate(over='raise', invalid='raise'):
                prediction = X_new @ self.coef_ + self.intercept_
            if not np.isfinite(prediction).all():
                raise ValueError('Predictions exceed float64 range.')
            return prediction
        except FloatingPointError as error:
            raise ValueError('Predictions exceed float64 range.') from error
