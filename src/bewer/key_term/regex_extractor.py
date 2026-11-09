"""Regex-based vocabulary extractors.

A :class:`RegexExtractor` is a reusable :data:`~bewer.key_term.vocabulary.ExtractorFn`: a
callable ``(dataset) -> Iterable[str]`` that scans each example's reference and hypothesis and
returns every token whose surface form fully matches a configured regular expression. It is the
generic building block behind metric-specific extractors such as
:data:`~bewer.key_term.orthographically_complex_term_extractor.ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN`.

To define a new family of regex-extracted terms, either instantiate ``RegexExtractor``
with a ``pattern`` directly, or subclass it and override :attr:`default_pattern`::

    class GreekLetterExtractor(RegexExtractor):
        default_pattern = r"\\p{Greek}+"

Either form can be registered as a vocabulary via
``Vocabulary(name, extractors=[extractor])`` and then referenced from a key-term metric.

Terms are always extracted from both sides. A pattern defines a *class* of terms, so a
hypothesis token in that class is a candidate key term whether or not any reference contains
it. Extracting from references only would hide hallucinated class terms from precision and
false-positive counts (a hypothesis ``HbA1c`` with no reference ``HbA1c`` would never be
matched).

Recall changes only when a hypothesis-only term matches reference tokens after normalization.
Terms are matched on normalized tokens, so under a lowercasing normalizer a hypothesis ``MRI``
also matches a reference ``mri`` that the pattern alone would not extract, adding a reference
occurrence. Under a case-preserving normalizer (e.g. ``cased``, used by the
orthographically-complex term metrics) a hypothesis-only term has no reference occurrences, and
recall is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import regex

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.text import Text, TokenList

__all__ = ["RegexExtractor", "match_token_regex"]


def match_token_regex(tokens: "TokenList", pattern: "regex.Pattern") -> list[slice]:
    """Locate tokens whose standardized surface form fully matches ``pattern``.

    Args:
        tokens: The tokens to scan.
        pattern: A compiled pattern each token's ``standardized`` form is full-matched against.

    Returns:
        A list of unit-length slices, one per matching token, so callers can reconstruct
        the matched span from the source text. Matching uses the case-preserving
        ``Token.standardized`` (not the normalized form), so patterns may key on case.
    """
    return [slice(i, i + 1) for i, token in enumerate(tokens) if pattern.fullmatch(token.standardized)]


class RegexExtractor:
    """Extract terms from a dataset's texts by full-matching tokens against a regex.

    Instances are :data:`~bewer.key_term.vocabulary.ExtractorFn` callables: ``extractor(dataset)``
    returns the set of distinct terms whose tokens fully match the pattern, ready to register via
    ``Vocabulary(name, extractors=[extractor])``. Both references and hypotheses are scanned (see
    the module docstring for why).

    Each text is read under the active tokenizer, so the surface form of a term
    depends on the pipeline the owning metric runs with (e.g. the ``orthographically_complex_term`` tokenizer
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
        """The compiled pattern each token is full-matched against."""
        return self._pattern

    def __call__(self, dataset: "Dataset") -> set[str]:
        """Return the set of distinct terms matching the pattern across ``dataset``'s references and hypotheses."""
        terms: set[str] = set()
        for example in dataset:
            for text in (example.ref, example.hyp):
                tokens = text.tokens
                terms.update(self._term_string(text, tokens, span) for span in match_token_regex(tokens, self._pattern))
        return terms

    @staticmethod
    def _term_string(text: "Text", tokens: "TokenList", span: slice) -> str:
        """Reconstruct a matched term from the standardized source, preserving its surface form."""
        return text.standardized[tokens[span.start].start : tokens[span.stop - 1].end]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._pattern.pattern!r})"
