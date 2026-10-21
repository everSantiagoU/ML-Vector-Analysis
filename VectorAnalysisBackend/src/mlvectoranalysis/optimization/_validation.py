"""Shared validation, performed before any complex-to-real conversion."""
from numbers import Integral, Real

import numpy as np

from .objective import ObjectiveEvaluationError


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def real(value, name, minimum=0.0, strict=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (value <= minimum if strict else value < minimum)):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {minimum}.")
    return float(value)


def real_array(value, name, ndim, block_size=8192):
    try:
        array = np.asarray(value)
        if array.ndim != ndim or array.dtype.kind not in 'iuf':
            raise ValueError(f"{name} must be a real numeric {ndim}D array (no object/complex/bool).")
        with np.errstate(over='raise', invalid='raise'):
            array = np.asarray(array, dtype=np.float64)
    except (TypeError, OverflowError, FloatingPointError) as error:
        raise ValueError(f"{name} cannot be represented as float64.") from error
    # Fixed-size element chunks, including non-contiguous inputs: no n*d or B*d mask.
    for chunk in np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                           op_flags=['readonly'], buffersize=min(block_size, 8192)):
        if not np.isfinite(chunk).all():
            raise ValueError(f"{name} must contain finite float64 values.")
    return array


def parameters(value, dimension):
    array = real_array(value, 'parameters', 1)
    if array.shape != (dimension,):
        raise ValueError(f"parameters must have shape ({dimension},).")
    return array


def finite_scalar(value):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in 'iuf' or not np.isfinite(array):
        raise ObjectiveEvaluationError("Objective must return a real finite scalar.")
    return float(array)


def norm(vector):
    # hypot avoids squaring large entries merely to compute a norm.
    result = float(np.hypot.reduce(vector, initial=0.0))
    if not np.isfinite(result):
        raise ObjectiveEvaluationError("Gradient norm exceeds float64 range.")
    return result
