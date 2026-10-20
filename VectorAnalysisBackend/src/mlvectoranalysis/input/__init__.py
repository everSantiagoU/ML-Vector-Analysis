from mlvectoranalysis.input.input_normalizer import InputNormalizer
from mlvectoranalysis.input.input_validator import InputValidator

__all__ = ["InputNormalizer", "InputValidator", "UserInputHandler"]


def __getattr__(name: str):
    # The parser uses this package; load the facade lazily to avoid a cycle.
    if name == "UserInputHandler":
        from mlvectoranalysis.input.user_input_handler import UserInputHandler

        return UserInputHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
