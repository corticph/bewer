from __future__ import annotations

from typing import TYPE_CHECKING

import regex as re

if TYPE_CHECKING:
    from bewer.core.text import TokenList

__all__ = ["ALPHANUM_DEFAULT_PATTERN", "match_token_regex", "tokens_are_hyphen_connected"]


ALPHANUM_DEFAULT_PATTERN = (
    # Branch 1: token (or hyphen-joined compound) has at least one uppercase letter
    # AND is not an "ordinary capitalised compound" (one or more parts that are each
    # either an init-cap word or an all-lowercase word, joined by hyphens). Catches
    # single tokens like MRI, mmHg, HbA1c, mRNA, eGFR, CH3, 3D, 5G, μM, ΔG, and
    # hyphen-joined compounds like CT-scan, X-ray, T-cell, pre-MRI, non-COVID,
    # 5-HT, MRI-CT, pre-COVID-19.
    r"(?=.*\p{Lu})"
    r"(?!(?:\p{Lu}\p{Ll}+|\p{Ll}+)(?:-(?:\p{Lu}\p{Ll}+|\p{Ll}+))*\Z)"
    r"[\p{L}\d][-\p{L}\d]*[\p{L}\d]"
    r"|"
    # Branch 2: letter(s) followed by at least one digit. Catches all-lowercase
    # letter+digit tokens (β2, o2, b12, hello1) so case-mismatches between ref and
    # hyp still register on the alignment as FN/FP.
    r"\p{L}+\d[\p{L}\d]*"
    r"|"
    # Branch 3: any token or compound containing at least one Greek letter
    # (μg, α, β2 already by branch 2, α-helix, β-blocker, γδ, ΔG already by branch 1).
    # Greek letters are always treated as entity-bearing regardless of case.
    r"(?=.*\p{Greek})[\p{L}\d][-\p{L}\d]*"
)


def match_token_regex(tokens: "TokenList", pattern: re.Pattern) -> list[slice]:
    """Return slices for tokens — or hyphen-connected token groups — matching the pattern.

    A run of consecutive tokens whose adjacent character gaps in the standardized
    source text consist of one or more hyphens (and nothing else) is treated as a
    compound candidate: the joined substring of the standardized text is matched
    against `pattern.fullmatch`. If it matches, the entire run is returned as one
    slice. Otherwise each token in the run is matched individually against
    `Token.raw` via `pattern.fullmatch`.
    """
    n = len(tokens)
    if n == 0:
        return []
    standardized = tokens[0].src.standardized if tokens[0].src is not None else None

    matches: list[slice] = []
    i = 0
    while i < n:
        # Extend the hyphen-connected run as far as possible.
        j = i
        if standardized is not None:
            while j + 1 < n and _hyphens_only(standardized, tokens[j].end, tokens[j + 1].start):
                j += 1
        # Compound match takes priority over per-token match within the run.
        if j > i:
            compound = standardized[tokens[i].start : tokens[j].end]
            if pattern.fullmatch(compound):
                matches.append(slice(i, j + 1))
                i = j + 1
                continue
        # Fall back to per-token matching across the run.
        for k in range(i, j + 1):
            if pattern.fullmatch(tokens[k].raw):
                matches.append(slice(k, k + 1))
        i = j + 1
    return matches


def tokens_are_hyphen_connected(tokens: "TokenList", start: int, stop: int) -> bool:
    """True iff each adjacent pair of tokens in `tokens[start:stop]` is joined in the
    standardized source text by one or more hyphens (and nothing else).

    Trivially returns True if the range has fewer than 2 tokens. Returns False if the
    tokens have no source attached (cannot inspect the gaps between them).
    """
    if stop - start < 2:
        return True
    src = tokens[start].src
    if src is None:
        return False
    text = src.standardized
    for k in range(start, stop - 1):
        if not _hyphens_only(text, tokens[k].end, tokens[k + 1].start):
            return False
    return True


def _hyphens_only(text: str, start: int, end: int) -> bool:
    """True iff `text[start:end]` is one or more hyphen characters and nothing else."""
    if end <= start:
        return False
    return all(c == "-" for c in text[start:end])
