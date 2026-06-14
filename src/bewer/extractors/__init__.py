"""Library of pre-defined vocabulary extractors.

Each extractor here is a :data:`~bewer.core.vocabulary.ExtractorFn` — a callable
``(dataset) -> Iterable[str]`` that derives key terms from a dataset on demand and can be
registered with ``Vocabulary(name).add_extractor(extractor)`` (then referenced from a
key-term metric). :class:`RegexExtractor` is the generic, reusable base for deriving terms
by full-matching tokens against a regular expression.
"""

from bewer.extractors.complex_term import COMPLEX_TERM_DEFAULT_PATTERN, ComplexTermExtractor
from bewer.extractors.regex import RegexExtractor, match_token_regex

__all__ = [
    "RegexExtractor",
    "match_token_regex",
    "ComplexTermExtractor",
    "COMPLEX_TERM_DEFAULT_PATTERN",
]
