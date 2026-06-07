"""Complex-term vocabulary extractor, with its regex matching primitive.

A :class:`ComplexTermExtractor` is a :data:`~bewer.core.vocabulary.VocabularyExtractor`:
called with a :class:`~bewer.core.dataset.Dataset`, it scans every example's *reference*
text for "complex terms" — abbreviations, acronyms, alphanumerics, hyphen compounds and
Greek-bearing tokens such as ``MRI``, ``HbA1c``, ``CO2``, ``3D`` and ``CT-scan`` — using
:data:`COMPLEX_TERM_DEFAULT_PATTERN`.

It returns a mapping of example index to the set of complex-term strings found in that
example's reference, i.e. a *local* vocabulary: each extracted term is associated with the
example it was found in. Registered via
:meth:`Dataset.add_vocabulary_from_function`, it backs the complex-term metrics (CTR / CTP
/ CTF).

The matching machinery — :func:`match_token_regex` and :data:`COMPLEX_TERM_DEFAULT_PATTERN`
— lives here alongside its sole consumer. Complex terms are tokenized with the
``complex_term`` pipeline, which does not split on hyphens, so a hyphenated compound such as
``CT-scan`` arrives as a single token; ``match_token_regex`` therefore just full-matches each
token against the pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import regex as re

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.text import Text, TokenList

__all__ = ["COMPLEX_TERM_DEFAULT_PATTERN", "ComplexTermExtractor", "match_token_regex"]


# --- COMPLEX_TERM_DEFAULT_PATTERN -------------------------------------------------------
#
# A token (or hyphen-joined compound, evaluated as one string with its hyphens) is a
# "complex term" when it shows at least one piece of *case-distinctive* or *alphanumeric*
# evidence. Each piece of evidence is a zero-width lookahead below; the body then consumes
# the whole string (the pattern is meant to be used with ``fullmatch``).
#
# Two families of evidence, with different reach:
#
# - Case-distinctive / Greek evidence (Greek letter, >=2 uppercase in a segment,
#   lowercase-then-uppercase, or a lone uppercase-letter segment) promotes the *whole*
#   token, including a hyphen-joined compound: "Hello-MRI" matches on the "MRI" segment.
#   These checks use ``[^-]*`` so a hyphen is a hard boundary — the evidence must lie within
#   one segment. That separates "CT-scan" (two uppercase in "CT") from "Hello-World" (one
#   uppercase per segment — an ordinary capitalised compound, rejected).
#
# - Alphanumeric evidence (a letter adjacent to a digit, e.g. "B12", "3D", "b12") needs no
#   segment bound: a digit is something ordinary prose never contains, so its presence is a
#   reliable signal that promotes the whole token, including a hyphen compound such as
#   "random-b12-text". Title-case phrases ("Hello-World") have no such signal and stay out.

_ALNUM = r"[\p{L}\p{N}]"

# The matched body: alphanumeric segments joined by single hyphens (no leading/trailing
# or doubled hyphen). With the complex_term tokenizer (no hyphen split), a hyphenated
# compound such as "CT-scan" is a single token and is fullmatched here whole.
_BODY = rf"{_ALNUM}+(?:-{_ALNUM}+)*"

# Case-distinctive / Greek evidence (compound-promoting).
_HAS_GREEK = r"(?=.*\p{Greek})"  # any Greek letter anywhere (e.g. μM, ΔG, α, γδ)
_TWO_UPPER = r"(?=.*\p{Lu}[^-]*\p{Lu})"  # >=2 uppercase in one segment (MRI, HbS, CT-scan)
_LOWER_UPPER = r"(?=.*\p{Ll}[^-]*\p{Lu})"  # lowercase then uppercase (mmHg, iPhone, mAb)
# A lone uppercase-letter segment inside a compound (X-ray, D-glucose, vitamin-D). The
# hyphen adjacency requirement excludes a standalone single capital ("A", "I"), which is
# an ordinary initial, not an entity.
_SOLO_UPPER_SEGMENT = r"(?=.*(?:(?:^|-)\p{Lu}-|-\p{Lu}(?:-|$)))"

# Alphanumeric evidence: a letter before a digit (B12, b12, hello1) or a digit before an
# uppercase letter (3D, 5HT). Unbounded by hyphens, so a digit anywhere promotes the whole
# token (random-b12-text). Digit-then-lowercase alone (the ordinal "1st") matches neither
# ordering and is rejected.
_ALNUM_MIX = r"(?:(?=.*\p{L}.*\p{Nd})|(?=.*\p{Nd}.*\p{Lu}))"

COMPLEX_TERM_DEFAULT_PATTERN = (
    rf"(?:{_HAS_GREEK}|{_TWO_UPPER}|{_LOWER_UPPER}|{_SOLO_UPPER_SEGMENT}|{_ALNUM_MIX}){_BODY}"
)


def match_token_regex(tokens: TokenList, pattern: re.Pattern) -> list[slice]:
    """Locate tokens fully matching ``pattern``.

    Matching uses each token's case-preserving ``raw`` text and ``re.Pattern.fullmatch``
    semantics, so only tokens whose entire text matches register. Complex terms are
    tokenized with the ``complex_term`` pipeline, which does not split on hyphens, so a
    hyphenated compound such as ``CT-scan`` arrives here as a single token and is matched
    whole — no cross-token grouping is needed.

    Args:
        tokens: The tokens to scan (offsets within the returned slices are positions in
            this list).
        pattern: A compiled regular expression to match token text against.

    Returns:
        A list of unit-length slices, one per matching token, in order of appearance.
    """
    return [slice(i, i + 1) for i, token in enumerate(tokens) if pattern.fullmatch(token.raw)]


class ComplexTermExtractor:
    """Extract complex terms from a dataset's reference texts.

    The extractor is a callable taking a :class:`Dataset` and returning a mapping of example
    index to the complex terms found in that example's reference. Terms are extracted from
    the case-preserving raw token text, so the returned strings keep their original casing
    (``MRI``, ``HbA1c``, ``CT-scan``); matching against hypotheses later applies the active
    normalizer as usual.

    It reads ``example.ref.tokens`` under the *active* tokenizer and expects one that does
    not split on hyphens (the ``complex_term`` pipeline the CT metrics register), so a
    hyphenated compound such as ``CT-scan`` is a single token. Under a hyphen-splitting
    tokenizer it would instead extract the qualifying fragments (``CT``) — run it within the
    ``complex_term`` pipeline context.

    Args:
        pattern: The pattern identifying complex terms. Defaults to
            :data:`COMPLEX_TERM_DEFAULT_PATTERN`. A string is compiled; a compiled pattern
            is used as-is.
    """

    def __init__(self, pattern: Union[str, re.Pattern] = COMPLEX_TERM_DEFAULT_PATTERN):
        self._pattern = re.compile(pattern) if isinstance(pattern, str) else pattern

    @property
    def pattern(self) -> re.Pattern:
        """The compiled pattern used to identify complex terms."""
        return self._pattern

    def __call__(self, dataset: Dataset) -> dict[int, set[str]]:
        """Return ``{example_index: {complex term, ...}}`` for examples with matches."""
        result: dict[int, set[str]] = {}
        for example in dataset:
            tokens = example.ref.tokens
            terms = {self._term_string(example.ref, tokens, span) for span in match_token_regex(tokens, self._pattern)}
            if terms:
                result[example.index] = terms
        return result

    @staticmethod
    def _term_string(text: Text, tokens: TokenList, span: slice) -> str:
        """Reconstruct a matched term from the standardized source, preserving its hyphens."""
        return text.standardized[tokens[span.start].start : tokens[span.stop - 1].end]

    def __repr__(self) -> str:
        return f"ComplexTermExtractor({self._pattern.pattern!r})"
