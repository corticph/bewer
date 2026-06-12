from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import ahocorasick

from bewer.core.text import Text, TextType, TokenList

if TYPE_CHECKING:
    from bewer.configs.resolve import Pipelines

__all__ = ["KeyTerm", "KeyTermMatch"]


class KeyTerm(Text):
    """A key term that can locate itself within reference text tokens.

    Inherits standardized, tokens, and pipeline caching from Text.
    Adds contiguous token matching against a reference TokenList.
    """

    def __init__(
        self,
        raw: str,
        *,
        pipelines: "Pipelines",
    ):
        super().__init__(raw=raw, pipelines=pipelines, src=None, text_type=TextType.KEY_TERM)

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'KeyTerm("{text}")'


@dataclass(frozen=True)
class KeyTermMatch:
    """A structured key term match: a token span in a ``Text`` identified as a ``KeyTerm``.

    Carries the matched token span (``start``/``stop`` token indices, mirroring the raw
    ``slice`` previously returned so existing consumers that read ``.start``/``.stop`` keep
    working) together with the parent ``text`` it was found in and the ``key_term`` it was
    identified as. Richer details — the side it was found on, the matched tokens, and the
    surface text — are derived from those two references.
    """

    start: int
    stop: int
    text: "Text"
    key_term: "KeyTerm"

    @property
    def token_slice(self) -> slice:
        """The matched span as a ``slice`` for indexing the parent text's ``TokenList``."""
        return slice(self.start, self.stop)

    @property
    def side(self) -> Optional[TextType]:
        """Which side the match was found on (``REF`` / ``HYP`` / ``KEY_TERM``)."""
        return self.text.text_type

    @property
    def tokens(self) -> TokenList:
        """The matched ``Token`` objects from the parent text."""
        return self.text.tokens[self.token_slice]

    @property
    def surface(self) -> str:
        """The raw surface text actually matched in the parent text."""
        return " ".join(self.tokens.raw)

    @property
    def term(self) -> str:
        """The raw string of the key term that matched."""
        return self.key_term.raw

    def __repr__(self):
        return f"KeyTermMatch(term={self.term!r}, span=({self.start}, {self.stop}), surface={self.surface!r})"


class KeyTermTrie:
    """Aho-Corasick automaton for efficient multi-key-term matching in token sequences."""

    def __init__(
        self,
        key_terms: set[KeyTerm],
        normalized: bool = True,
        add_capitalized: bool = False,
    ):
        """Initialize the automaton with the given set of key terms.

        Args:
            key_terms: A set of KeyTerm objects to build the automaton from.
            normalized: Whether to use normalized text for matching. If False, uses raw text.
            add_capitalized: Whether to add capitalized versions of key terms for case-insensitive matching.
                Only applies if normalized is False and is only applied to the first word in an n-gram key term.
        """
        self.normalized = normalized
        self.add_capitalized = add_capitalized

        # (key_term, token_pattern) pairs, keeping each pattern tied to its originating
        # key term so matches can report which KeyTerm they were identified as.
        pattern_entries: list[tuple[KeyTerm, tuple[str, ...]]] = []
        for key_term in key_terms:
            tokens = key_term.tokens.normalized if normalized else key_term.tokens.raw
            token_pattern = tuple(tokens)
            if not token_pattern:
                continue
            pattern_entries.append((key_term, token_pattern))

        # Handle capitalization variants, mapped back to their originating key term.
        if add_capitalized and not normalized:
            for key_term, p in list(pattern_entries):
                first_cap = p[0].capitalize()
                if first_cap != p[0]:
                    pattern_entries.append((key_term, (first_cap,) + p[1:]))

        # Build vocab: token string -> int for KEY_SEQUENCE mode
        self._vocab = {w: i for i, w in enumerate({w for _, p in pattern_entries for w in p})}
        self._unknown = len(self._vocab)

        # Build Aho-Corasick automaton; payload carries (pattern_len, key_term).
        self._automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_SEQUENCE)
        seen = set()
        for key_term, pattern in pattern_entries:
            int_pattern = tuple(self._vocab[w] for w in pattern)
            if int_pattern not in seen:
                self._automaton.add_word(int_pattern, (len(pattern), key_term))
                seen.add(int_pattern)
        self._automaton.make_automaton()

    def encode(self, tokens: TokenList) -> tuple[int, ...]:
        """Encode a token list into the trie's integer vocabulary."""
        token_strings = tokens.normalized if self.normalized else tokens.raw
        return tuple(self._vocab.get(w, self._unknown) for w in token_strings)

    def find_in_tokens(self, tokens: TokenList) -> tuple[list[slice], list[KeyTerm]]:
        """Find all key term matches, returning token spans and the matched key terms."""
        int_text = self.encode(tokens)
        matches: list[slice] = []
        key_terms: list[KeyTerm] = []
        for end_idx, (pattern_len, key_term) in self._automaton.iter(int_text):
            start = end_idx - pattern_len + 1
            matches.append(slice(start, end_idx + 1))
            key_terms.append(key_term)
        return matches, key_terms


def _remove_duplicate_matches(matches: list[KeyTermMatch]) -> list[KeyTermMatch]:
    """Remove exact duplicate matches, preserving order."""
    seen: set[tuple[int, int]] = set()
    result = []
    for m in matches:
        key = (m.start, m.stop)
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _remove_subset_matches(matches: list[KeyTermMatch]) -> list[KeyTermMatch]:
    """Remove matches that are subsets of other matches, preferring longer matches."""
    if not matches:
        return matches
    # Sort by start ascending, then by length descending
    matches.sort(key=lambda s: (s.start, s.start - s.stop))
    result = [matches[0]]
    for m in matches[1:]:
        prev = result[-1]
        # Skip if fully contained within previous match
        if m.start >= prev.start and m.stop <= prev.stop:
            continue
        result.append(m)
    return result
