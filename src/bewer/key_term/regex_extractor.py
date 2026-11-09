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

Terms are extracted from both sides by default. A pattern defines a *class* of terms, so a
hypothesis token in that class is a candidate key term whether or not any reference contains
it. Extracting from references only would hide hallucinated class terms from precision and
false-positive counts (a hypothesis ``HbA1c`` with no reference ``HbA1c`` would never be
matched).

Recall changes only when a hypothesis-only term matches reference tokens after normalization.
Terms are matched on normalized tokens, so under a lowercasing normalizer a hypothesis ``MRI``
also matches a reference ``mri`` that the pattern alone would not extract, adding a reference
occurrence. Under a case-preserving normalizer (e.g. ``cased``, used by the
orthographically-complex term metrics) a hypothesis-only term has no reference occurrences, and
recall is unchanged. Pass ``sources=("ref",)`` to keep the reference-only behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import regex

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.text import Text, TokenList

__all__ = ["RegexExtractor", "match_token_regex"]

Source = Literal["ref", "hyp"]
_SOURCES: tuple[Source, ...] = ("ref", "hyp")


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
    ``Vocabulary(name, extractors=[extractor])``. By default both references and hypotheses are
    scanned (see the module docstring for why); pass ``sources=("ref",)`` to scan references only.

    Each text is read under the active tokenizer, so the surface form of a term
    depends on the pipeline the owning metric runs with (e.g. the ``orthographically_complex_term`` tokenizer
    keeps ``CT-scan`` a single token, while ``key_term`` splits it on the hyphen).
    """

    #: Pattern used when none is passed to ``__init__``. Subclasses override this; the base
    #: default matches nothing.
    default_pattern: Union[str, "regex.Pattern"] = r"(?!)"

    def __init__(
        self,
        pattern: Optional[Union[str, "regex.Pattern"]] = None,
        *,
        sources: Sequence[Source] = _SOURCES,
    ):
        """Initialize the extractor.

        Args:
            pattern: The pattern (string or compiled) to full-match tokens against. Defaults
                to :attr:`default_pattern`.
            sources: Which sides of each example to scan: any non-empty combination of
                ``"ref"`` and ``"hyp"``. Defaults to both.

        Raises:
            ValueError: If ``sources`` is empty or contains anything but ``"ref"`` and ``"hyp"``.
        """
        sources = tuple(dict.fromkeys(sources))
        if not sources or any(source not in _SOURCES for source in sources):
            raise ValueError(f"sources must be a non-empty combination of {_SOURCES}, got {sources!r}")
        pattern = self.default_pattern if pattern is None else pattern
        self._pattern = regex.compile(pattern) if isinstance(pattern, str) else pattern
        self._sources: tuple[Source, ...] = sources

    @property
    def pattern(self) -> "regex.Pattern":
        """The compiled pattern each token is full-matched against."""
        return self._pattern

    @property
    def sources(self) -> tuple[Source, ...]:
        """The sides of each example that are scanned (``"ref"`` and/or ``"hyp"``)."""
        return self._sources

    def __call__(self, dataset: "Dataset") -> set[str]:
        """Return the set of distinct terms matching the pattern across ``dataset``'s scanned sources."""
        terms: set[str] = set()
        for example in dataset:
            for source in self._sources:
                text = getattr(example, source)
                tokens = text.tokens
                terms.update(self._term_string(text, tokens, span) for span in match_token_regex(tokens, self._pattern))
        return terms

    @staticmethod
    def _term_string(text: "Text", tokens: "TokenList", span: slice) -> str:
        """Reconstruct a matched term from the standardized source, preserving its surface form."""
        return text.standardized[tokens[span.start].start : tokens[span.stop - 1].end]

    def __repr__(self) -> str:
        if self._sources == _SOURCES:
            return f"{type(self).__name__}({self._pattern.pattern!r})"
        return f"{type(self).__name__}({self._pattern.pattern!r}, sources={self._sources!r})"
