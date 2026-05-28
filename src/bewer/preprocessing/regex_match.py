from __future__ import annotations

from typing import TYPE_CHECKING

import regex as re

if TYPE_CHECKING:
    from bewer.core.text import TokenList

__all__ = ["ALPHANUM_DEFAULT_PATTERN", "match_token_regex"]


ALPHANUM_DEFAULT_PATTERN = (
    # Token contains an uppercase letter AND is not an ordinary capitalised word
    # (e.g. Patient, Hello). Catches MRI, HbA1c, mmHg, iPhone, mRNA, ΔG, μM, 5G, 3D,
    # CO2, HbS, eGFR, ΔΩ, etc.
    r"(?=.*\p{Lu})(?!\p{Lu}\p{Ll}+\Z)[\p{L}\d]{2,}"
    r"|"
    # All-lowercase or mixed-script letter+digit tokens (β2, o2, b12, hello1, ...).
    # Without an uppercase signal these are caught here so case-mismatches between
    # ref and hyp still register as FN/FP via the alignment.
    r"\p{L}+\d[\p{L}\d]*"
)


def match_token_regex(tokens: "TokenList", pattern: re.Pattern) -> list[slice]:
    """Return one slice(i, i+1) per token whose raw form matches the pattern.

    Matching is performed against `Token.raw` (case-preserving, post-standardization,
    pre-normalization) using `pattern.fullmatch` — so the predicate applies to the
    whole token, not a substring.
    """
    return [slice(i, i + 1) for i, token in enumerate(tokens) if pattern.fullmatch(token.raw)]
