r"""Orthographically-complex term extraction.

An *orthographically-complex term* is a token whose written surface form carries
case-distinctive or alphanumeric evidence that it is an abbreviation, acronym, alphanumeric
code or hyphen compound rather than an ordinary word — e.g. ``MRI``, ``mmHg``, ``HbA1c``,
``CO2``, ``CT-scan`` or ``α-helix``. :class:`OrthographicallyComplexTermExtractor` is a
:class:`~bewer.extractors.regex.RegexExtractor` that harvests such terms from a dataset's
references; :data:`ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN` backs the predefined
``orthographically_complex_term_recall`` / ``_precision`` / ``_fscore`` metrics.

It selects a term by its *orthography* (capitalization, digit/letter mixing, symbols, Greek),
not its meaning — so it is one dimension of "term complexity" (cf. a future
``phonetically_complex_term`` etc.). The default pattern accepts a token when its body —
alphanumeric segments joined by single hyphens — also satisfies at least one piece of
evidence:

* a Greek letter anywhere (``ΔG``, ``α-helix``),
* two uppercase letters within one hyphen segment (``MRI``, ``CT-scan``),
* a lowercase letter followed by an uppercase one within a segment (``mmHg``, ``iPhone``),
* a lone uppercase-letter segment in a compound (``X-ray``, ``D-glucose``), or
* a letter adjacent to a digit / a digit followed by an uppercase letter (``CO2``, ``B12``, ``5HT``).

It is matched case-sensitively against the raw token, so ordinary words (``Patient``),
ordinals (``1st``) and plain numbers (``2024``) are rejected.
"""

from __future__ import annotations

from bewer.extractors.regex import RegexExtractor

__all__ = ["ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN", "OrthographicallyComplexTermExtractor"]

# An alphanumeric character (any Unicode letter or number).
_ALNUM = r"[\p{L}\p{N}]"
# The matched span: alphanumeric segments joined by single hyphens (e.g. "CT-scan", "5-HT-receptor").
_BODY = rf"{_ALNUM}+(?:-{_ALNUM}+)*"

# Evidence lookaheads. Case-distinctive evidence is constrained to a single hyphen segment
# (``[^-]*``) so that e.g. "CT" in "CT-scan" qualifies, while "Patient-Care" does not.
_HAS_GREEK = r"(?=.*\p{Greek})"
_TWO_UPPER = r"(?=.*\p{Lu}[^-]*\p{Lu})"
_LOWER_UPPER = r"(?=.*\p{Ll}[^-]*\p{Lu})"
_SOLO_UPPER_SEGMENT = r"(?=.*(?:(?:^|-)\p{Lu}-|-\p{Lu}(?:-|$)))"
# Alphanumeric evidence is not segment-bound: a digit anywhere alongside a letter promotes
# the whole token.
_ALNUM_MIX = r"(?:(?=.*\p{L}.*\p{Nd})|(?=.*\p{Nd}.*\p{Lu}))"

ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN = (
    rf"(?:{_HAS_GREEK}|{_TWO_UPPER}|{_LOWER_UPPER}|{_SOLO_UPPER_SEGMENT}|{_ALNUM_MIX}){_BODY}"
)


class OrthographicallyComplexTermExtractor(RegexExtractor):
    """Extract orthographically-complex terms — abbreviations, acronyms, alphanumerics and
    hyphen compounds — from a dataset's reference texts.

    Defaults to :data:`ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN`. Expects the
    ``complex_term`` tokenizer (which does not split on hyphens), so a compound like ``CT-scan``
    arrives as a single token and is matched whole.
    """

    default_pattern = ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN
