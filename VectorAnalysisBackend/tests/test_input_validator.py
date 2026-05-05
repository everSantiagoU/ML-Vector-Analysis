import pytest

from mlvectoranalysis.input.input_validator import InputValidator


def test_input_validator_validate_format_with_angle_brackets():
    validator = InputValidator()

    assert validator.validate_format("<t,t**2,sin(t)>")


def test_input_validator_validate_format_with_named_vector_expression():
    validator = InputValidator()

    assert validator.validate_format("r(t)=<t,t**2,sin(t)>")


def test_input_validator_validate_format_rejects_missing_vector_delimiters():
    validator = InputValidator()

    assert not validator.validate_format("t,t**2,sin(t)")


def test_input_validator_extract_components_from_angle_brackets():
    validator = InputValidator()

    components = validator.extract_components("<t,t**2,sin(t)>")

    assert components == ("t", "t**2", "sin(t)")


def test_input_validator_extract_components_from_named_vector_expression():
    validator = InputValidator()

    components = validator.extract_components("r(t)=< t, t^2, sen(t) >")

    assert components == ("t", "t**2", "sin(t)")


def test_input_validator_extract_components_keeps_nested_commas_inside_functions():
    validator = InputValidator()

    components = validator.extract_components("<t,log(t),sqrt(t+1)>")

    assert components == ("t", "log(t)", "sqrt(t+1)")


def test_input_validator_extract_components_raises_for_empty_component():
    validator = InputValidator()

    with pytest.raises(ValueError):
        validator.extract_components("<t,,sin(t)>")


def test_input_validator_validate_components_accepts_three_valid_components():
    validator = InputValidator()

    assert validator.validate_components("<t,t**2,sin(t)>")


def test_input_validator_validate_components_rejects_wrong_component_count():
    validator = InputValidator()

    assert not validator.validate_components("<t,t**2>")


def test_input_validator_validate_components_rejects_invalid_sympy_expression():
    validator = InputValidator()

    assert not validator.validate_components("<t,t**,sin(t)>")


def test_input_validator_validate_allowed_symbols_accepts_supported_functions_and_constants():
    validator = InputValidator()

    assert validator.validate_allowed_symbols("<sqrt(t),exp(t),pi>")


def test_input_validator_validate_allowed_symbols_rejects_unknown_variable():
    validator = InputValidator()

    assert not validator.validate_allowed_symbols("<t,x,sin(t)>")


def test_input_validator_validate_allowed_symbols_accepts_custom_variable():
    validator = InputValidator(variable="x")

    assert validator.validate_allowed_symbols("<x,x**2,sin(x)>")


def test_input_validator_validate_component_expression_accepts_valid_component():
    validator = InputValidator()

    assert validator.validate_component_expression("t^2 + sen(t)")


def test_input_validator_validate_component_expression_rejects_invalid_component():
    validator = InputValidator()

    assert not validator.validate_component_expression("x + 1")


def test_input_validator_can_be_imported_from_input_package():
    from mlvectoranalysis.input import InputValidator as ExportedInputValidator

    assert ExportedInputValidator is InputValidator
