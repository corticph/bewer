"""Regex-based vocabulary extractors.

A :class:`RegexExtractor` is a reusable :data:`~bewer.core.vocabulary.ExtractorFn`: a
callable ``(dataset) -> Iterable[str]`` that scans each example's *reference* and returns
the terms a regular expression identifies. It is the generic building block behind
metric-specific patterns such as
:data:`~bewer.extractors.orthographically_complex_term.ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN`
and the quantity category patterns in :mod:`bewer.extractors.quantity`.

It supports two matching strategies, selected by the ``span`` flag:

* ``span=False`` (default) full-matches the pattern against each individual token's surface
  form — use it for single-token classes (acronyms, alphanumerics, bare numbers).
* ``span=True`` searches the pattern over the whole standardized text and maps each match
  back to the token span it covers — use it for classes that straddle token boundaries
  (e.g. ``5 mg``, ``95 %``, ``$ 100``), where the unit/symbol tokenizes off the number.

Instantiate it directly with a ``pattern`` (and optionally ``span``), or subclass it and
set :attr:`default_pattern` / :attr:`span`::

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

__all__ = ["RegexExtractor", "match_token_regex", "match_span_regex"]


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


def match_span_regex(text: "Text", pattern: "regex.Pattern") -> list[slice]:
    """Locate token spans whose region of the standardized text matches ``pattern``.

    The pattern is searched (``finditer``) over ``text.standardized``, and each non-empty
    match is mapped to the slice of tokens it overlaps. This captures terms that straddle
    token boundaries — e.g. ``5 mg`` or ``95 %``, where the number and the unit/symbol are
    separate tokens. Matching is case-preserving (the standardized text is not normalized),
    so patterns may key on case.

    Args:
        text: The text whose standardized form is searched and whose tokens are mapped to.
        pattern: A compiled pattern searched over the standardized text.

    Returns:
        A list of token slices, one per match, in match order.
    """
    tokens = text.tokens
    standardized = text.standardized
    spans: list[slice] = []
    for match in pattern.finditer(standardized):
        start, end = match.span()
        if start == end:
            continue
        covered = [i for i, token in enumerate(tokens) if token.start < end and token.end > start]
        if covered:
            spans.append(slice(covered[0], covered[-1] + 1))
    return spans


class RegexExtractor:
    """Extract reference terms identified by a regular expression.

    Instances are :data:`~bewer.core.vocabulary.ExtractorFn` callables: ``extractor(dataset)``
    returns the set of distinct reference terms matching the pattern, ready to register via
    ``Vocabulary(name).add_extractor(extractor)``.

    Each example's reference is read under the active tokenizer, so the surface form of a term
    depends on the pipeline the owning metric runs with (e.g. the ``orthographically_complex_term`` tokenizer
    keeps ``CT-scan`` a single token, while ``key_term`` splits it on the hyphen).

    The ``span`` flag selects the matching strategy (see the module docstring): per-token
    full-match (``False``) or whole-text search mapped to token spans (``True``).
    """

    #: Pattern used when none is passed to ``__init__``. Subclasses override this; the base
    #: default matches nothing.
    default_pattern: Union[str, "regex.Pattern"] = r"(?!)"

    #: Default matching strategy when ``span`` is not passed to ``__init__``.
    span: bool = False

    def __init__(self, pattern: Optional[Union[str, "regex.Pattern"]] = None, *, span: Optional[bool] = None):
        """Initialize the extractor.

        Args:
            pattern: The pattern (string or compiled) to match against. Defaults to
                :attr:`default_pattern`.
            span: Whether to match across token boundaries (whole-text search) rather than
                per token. Defaults to the class :attr:`span`.
        """
        pattern = self.default_pattern if pattern is None else pattern
        self._pattern = regex.compile(pattern) if isinstance(pattern, str) else pattern
        self._span = self.span if span is None else span

    @property
    def pattern(self) -> "regex.Pattern":
        """The compiled pattern reference text is matched against."""
        return self._pattern

    def _match_spans(self, text: "Text") -> list[slice]:
        """Return the token spans matched in ``text`` under the configured strategy."""
        if self._span:
            return match_span_regex(text, self._pattern)
        return match_token_regex(text.tokens, self._pattern)

    def __call__(self, dataset: "Dataset") -> set[str]:
        """Return the set of distinct reference terms matching the pattern across ``dataset``."""
        terms: set[str] = set()
        for example in dataset:
            text = example.ref
            tokens = text.tokens
            terms.update(self._term_string(text, tokens, span) for span in self._match_spans(text))
        return terms

    @staticmethod
    def _term_string(text: "Text", tokens: "TokenList", span: slice) -> str:
        """Reconstruct a matched term from the standardized source, preserving its surface form."""
        return text.standardized[tokens[span.start].start : tokens[span.stop - 1].end]

    def __repr__(self) -> str:
        span = ", span=True" if self._span else ""
        return f"{type(self).__name__}({self._pattern.pattern!r}{span})"
