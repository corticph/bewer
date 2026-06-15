"""Library of pre-defined vocabulary extractors and the regex term patterns behind them.

The core building block is :class:`RegexExtractor` — a :data:`~bewer.core.vocabulary.ExtractorFn`
(``(dataset) -> Iterable[str]``) that derives reference terms from a regular expression and is
registered via ``Vocabulary(name).add_extractor(extractor)`` (then referenced from a key-term
metric). The pattern constants here (complex-term and the quantity categories) are the inputs
to that extractor and to :func:`bewer.metrics.register_regex_term_metrics`.
"""

from bewer.extractors.orthographically_complex_term import (
    ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN,
    OrthographicallyComplexTermExtractor,
)
from bewer.extractors.quantity import (
    CURRENCY,
    CURRENCY_PATTERN,
    DEGREE,
    DEGREE_PATTERN,
    MEASUREMENT_PATTERN,
    NUMBER,
    NUMBER_PATTERN,
    PERCENT,
    PERCENTAGE_PATTERN,
    QUANTITY_CATEGORIES,
    UNIT,
)
from bewer.extractors.regex import RegexExtractor, match_span_regex, match_token_regex

__all__ = [
    "RegexExtractor",
    "match_token_regex",
    "match_span_regex",
    "OrthographicallyComplexTermExtractor",
    "ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN",
    # quantity components
    "NUMBER",
    "PERCENT",
    "DEGREE",
    "CURRENCY",
    "UNIT",
    # quantity category patterns
    "NUMBER_PATTERN",
    "PERCENTAGE_PATTERN",
    "DEGREE_PATTERN",
    "CURRENCY_PATTERN",
    "MEASUREMENT_PATTERN",
    "QUANTITY_CATEGORIES",
]
