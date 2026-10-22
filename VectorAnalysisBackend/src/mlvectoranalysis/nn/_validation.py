"""Float64 policy and explicit numeric failures for the NumPy network."""
from numbers import Integral, Real
import numpy as np

DTYPE = np.float64


class NumericalError(FloatingPointError):
    """An otherwise valid computation exceeded finite float64 arithmetic."""


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite real.')
    return float(value)


def array(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or (ndim is not None and raw.ndim != ndim):
        raise ValueError(f'{name} must be a real numeric {ndim or "N"}D array.')
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.asarray(raw, dtype=DTYPE)
    if 0 in result.shape:
        raise ValueError(f'{name} must be nonempty and finite in float64.')
    # Bound validation temporaries for resident datasets; dtype conversion above
    # can still require a full one-time copy and is part of the memory contract.
    blocks = (result[start:start + 8192] for start in range(0, len(result), 8192)) if result.ndim else (result,)
    if any(not np.isfinite(block).all() for block in blocks):
        raise ValueError(f'{name} must be nonempty and finite in float64.')
    return result


def finite(value):
    if not np.isfinite(value).all():
        raise NumericalError('Computation exceeds finite float64 range.')
    return value


def numeric(function):
    """Translate NumPy arithmetic overflow, including kernels checked by finite()."""
    from functools import wraps

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError) as error:
            raise NumericalError(f'{function.__name__}: computation exceeds finite float64 range.') from error
    return wrapped
