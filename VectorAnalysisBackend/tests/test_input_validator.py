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


def test_invalid_vectors_rejected(invalid_vector):
    assert InputValidator().validate_components(invalid_vector) is False


def test_invalid_scalar_grammar(invalid_component):
    validator = InputValidator()
    assert validator.validate_component_expression(invalid_component) is False
    if isinstance(invalid_component, str):
        assert validator.validate_allowed_symbols(f'<{invalid_component},t,t>') is False


@pytest.mark.parametrize('text', ['r(x)=<t,t,t>', 'prefix=<t,t,t>', 'r(t)=r(t)=<t,t,t>', '<(t,t,t>', '<t),t,t>', None])
def test_bad_format_and_extraction(text):
    validator = InputValidator()
    assert validator.validate_format(text) is False
    with pytest.raises(ValueError):
        validator.extract_components(text)


def test_genuinely_nested_comma():
    validator = InputValidator()
    assert validator.extract_components('<log(t,2),t,0>') == ('log(t,2)', 't', '0')
    assert validator.validate_components('<log(t,2),t,0>')


@pytest.mark.parametrize('method', ['validate_format', 'validate_components', 'validate_allowed_symbols', 'validate_component_expression'])
def test_boolean_methods_handle_nontext(method):
    assert getattr(InputValidator(), method)(None) is False


def test_unsupported_tree_rejected_before_symbolic_construction(monkeypatch):
    import mlvectoranalysis._scalar_expression as grammar

    def forbidden(*args):
        pytest.fail('Symbolic construction ran before grammar validation finished')

    monkeypatch.setattr(grammar, '_build', forbidden)
    assert not InputValidator().validate_component_expression('sin(t)+t.real')


def test_expected_components_remains_configurable():
    validator = InputValidator(expected_components=2)
    assert validator.validate_components('<t,t>')
    assert not validator.validate_components('<t,t,t>')
