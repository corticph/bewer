"""Regex-based vocabulary extractors.

A :class:`RegexExtractor` is a reusable :data:`~bewer.core.vocabulary.ExtractorFn`: a
callable ``(dataset) -> Iterable[str]`` that scans each example's *reference* and returns
every token whose surface form fully matches a configured regular expression. It is the
generic building block behind metric-specific extractors such as
:data:`~bewer.extractors.orthographically_complex_term.ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN`.

To define a new family of regex-extracted terms, either instantiate ``RegexExtractor``
with a ``pattern`` directly, or subclass it and override :attr:`default_pattern`::

    class GreekLetterExtractor(RegexExtractor):
        default_pattern = r"\\p{Greek}+"

Either form can be registered as a vocabulary via
``Vocabulary(name).add_extractor(extractor)`` and then referenced from a key-term metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import regex

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.text import Text, TokenList

__all__ = ["RegexExtractor", "match_token_regex"]


def match_token_regex(tokens: "TokenList", pattern: "regex.Pattern") -> list[slice]:
    """Locate tokens whose raw surface form fully matches ``pattern``.

    Args:
        tokens: The tokens to scan.
        pattern: A compiled pattern each token's ``raw`` form is full-matched against.

    Returns:
        A list of unit-length slices, one per matching token, so callers can reconstruct
        the matched span from the source text. Matching uses the case-preserving
        ``Token.raw`` (not the normalized form), so patterns may key on case.
    """
    return [slice(i, i + 1) for i, token in enumerate(tokens) if pattern.fullmatch(token.raw)]


class RegexExtractor:
    """Extract terms from a dataset's reference texts by full-matching tokens against a regex.

    Instances are :data:`~bewer.core.vocabulary.ExtractorFn` callables: ``extractor(dataset)``
    returns the set of distinct reference terms whose tokens fully match the pattern, ready to
    register via ``Vocabulary(name).add_extractor(extractor)``.

    Each example's reference is read under the active tokenizer, so the surface form of a term
    depends on the pipeline the owning metric runs with (e.g. the ``complex_term`` tokenizer
    keeps ``CT-scan`` a single token, while ``key_term`` splits it on the hyphen).
    """

    #: Pattern used when none is passed to ``__init__``. Subclasses override this; the base
    #: default matches nothing.
    default_pattern: Union[str, "regex.Pattern"] = r"(?!)"

    def __init__(self, pattern: Optional[Union[str, "regex.Pattern"]] = None):
        """Initialize the extractor.

        Args:
            pattern: The pattern (string or compiled) to full-match tokens against. Defaults
                to :attr:`default_pattern`.
        """
        pattern = self.default_pattern if pattern is None else pattern
        self._pattern = regex.compile(pattern) if isinstance(pattern, str) else pattern

    @property
    def pattern(self) -> "regex.Pattern":
        """The compiled pattern each reference token is full-matched against."""
        return self._pattern

    def __call__(self, dataset: "Dataset") -> set[str]:
        """Return the set of distinct reference terms matching the pattern across ``dataset``."""
        terms: set[str] = set()
        for example in dataset:
            text = example.ref
            tokens = text.tokens
            terms.update(self._term_string(text, tokens, span) for span in match_token_regex(tokens, self._pattern))
        return terms

    @staticmethod
    def _term_string(text: "Text", tokens: "TokenList", span: slice) -> str:
        """Reconstruct a matched term from the standardized source, preserving its surface form."""
        return text.standardized[tokens[span.start].start : tokens[span.stop - 1].end]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._pattern.pattern!r})"
