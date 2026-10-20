from mlvectoranalysis.analysis.analysis_result import AnalysisResult
from mlvectoranalysis.analysis.vector_analyzer import VectorAnalyzer
from mlvectoranalysis.input.input_normalizer import InputNormalizer
from mlvectoranalysis.input.input_validator import InputValidator
from mlvectoranalysis.parsing.expression_parser import ExpressionParser


class UserInputHandler:
    """Receive external text; delegate validation and construction to the parser."""

    def __init__(
        self,
        variable: str = "t",
        normalizer: InputNormalizer | None = None,
        validator: InputValidator | None = None,
        parser: ExpressionParser | None = None,
        analyzer: VectorAnalyzer | None = None,
    ) -> None:
        self.raw_text = ""
        self.variable = variable
        if parser is not None:
            if parser.variable != variable:
                raise ValueError("Handler and parser must use the same variable.")
            if validator is not None and validator is not parser.validator:
                raise ValueError("Injected parser must use the supplied validator.")
            if normalizer is not None and normalizer is not parser.normalizer:
                raise ValueError("Injected parser must use the supplied normalizer.")
            validator = parser.validator
            normalizer = parser.normalizer
        self.normalizer = normalizer if normalizer is not None else (
            validator.normalizer if validator is not None else InputNormalizer()
        )
        self.validator = validator if validator is not None else InputValidator(
            variable=variable, normalizer=self.normalizer,
        )
        self.parser = parser if parser is not None else ExpressionParser(
            variable=variable, normalizer=self.normalizer, validator=self.validator,
        )
        self.analyzer = analyzer if analyzer is not None else VectorAnalyzer()

    def receive_input(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        self.raw_text = text
        return text

    def process_input(self) -> AnalysisResult:
        if not self.raw_text.strip():
            raise ValueError("Receive nonempty input before processing.")
        normalized_input = self.normalizer.normalize(self.raw_text)
        vector = self.parser.parse_vector_expression(normalized_input)
        return self.analyzer.analyze(vector, normalized_input=normalized_input)
