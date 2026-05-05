from mlvectoranalysis.input.input_normalizer import InputNormalizer


def test_input_normalizer_remove_extra_spaces():
    normalizer = InputNormalizer()

    assert normalizer.remove_extra_spaces("  < t ,  t^2 ,  sin(t) >  ") == "<t,t^2,sin(t)>"


def test_input_normalizer_standardize_power_operator():
    normalizer = InputNormalizer()

    assert normalizer.standardize_syntax("t^2+1") == "t**2+1"


def test_input_normalizer_standardize_spanish_trigonometric_function():
    normalizer = InputNormalizer()

    assert normalizer.standardize_syntax("sen(t)+tg(t)") == "sin(t)+tan(t)"


def test_input_normalizer_standardize_square_root_and_pi():
    normalizer = InputNormalizer()

    assert normalizer.standardize_syntax("√(t)+π") == "sqrt(t)+pi"


def test_input_normalizer_normalize_combines_spacing_and_syntax_rules():
    normalizer = InputNormalizer()

    normalized_text = normalizer.normalize("  r(t) = < sen(t) , t^2 , √(t) >  ")

    assert normalized_text == "r(t)=<sin(t),t**2,sqrt(t)>"


def test_input_normalizer_can_be_imported_from_input_package():
    from mlvectoranalysis.input import InputNormalizer as ExportedInputNormalizer

    assert ExportedInputNormalizer is InputNormalizer
