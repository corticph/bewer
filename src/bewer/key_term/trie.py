from __future__ import annotations

from typing import TYPE_CHECKING

import ahocorasick

from bewer.core.text import TokenList
from bewer.key_term.key_term import KeyTerm, KeyTermMatch

if TYPE_CHECKING:
    from bewer.core.text import Text

__all__ = ["KeyTermTrie"]


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
            normalized: Whether to use normalized text for matching. If False, uses standardized text.
            add_capitalized: Whether to add capitalized versions of key terms for case-insensitive matching.
                Only applies if normalized is False and is only applied to the first word in an n-gram key term.
        """
        self.normalized = normalized
        self.add_capitalized = add_capitalized

        # The input terms, kept for introspection (e.g. multi-token term detection).
        self._key_terms = frozenset(key_terms)

        # (key_term, token_pattern) pairs, keeping each pattern tied to its originating
        # key term so matches can report which KeyTerm they were identified as. Iterate in
        # a stable order (by raw text) so that when distinct key terms collapse to the same
        # token pattern, the first-wins de-duplication below picks the same one every run.
        pattern_entries: list[tuple[KeyTerm, tuple[str, ...]]] = []
        for key_term in sorted(key_terms, key=lambda kt: kt.raw):
            tokens = key_term.tokens.normalized if normalized else key_term.tokens.standardized
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

    @property
    def key_terms(self) -> frozenset[KeyTerm]:
        """The key terms this trie was built from.

        Includes every input term, also when distinct terms collapse to the same
        token pattern (first-wins de-duplication in the automaton drops the duplicate
        pattern, not the term here).
        """
        return self._key_terms

    def encode(self, tokens: TokenList) -> tuple[int, ...]:
        """Encode a token list into the trie's integer vocabulary."""
        token_strings = tokens.normalized if self.normalized else tokens.standardized
        return tuple(self._vocab.get(w, self._unknown) for w in token_strings)

    def find_in_tokens(self, tokens: TokenList) -> tuple[list[slice], list[KeyTerm]]:
        """Find all key term matches, returning token spans and the matched key terms."""
        if not self._vocab:
            return [], []
        int_text = self.encode(tokens)
        matches: list[slice] = []
        key_terms: list[KeyTerm] = []
        for end_idx, (pattern_len, key_term) in self._automaton.iter(int_text):
            start = end_idx - pattern_len + 1
            matches.append(slice(start, end_idx + 1))
            key_terms.append(key_term)
        return matches, key_terms

    def find_matches(
        self,
        text: "Text",
        *,
        allow_subset_matches: bool = False,
    ) -> list[KeyTermMatch]:
        """Find key term matches in ``text``'s tokens and wrap them as ``KeyTermMatch`` objects.

        Args:
            text: The :class:`~bewer.core.text.Text` to search.
            allow_subset_matches: If False (default), discard matches that are subsets
                of longer matches. If True, keep all matches but deduplicate exact overlaps.

        Returns:
            A list of :class:`KeyTermMatch` objects.
        """
        spans, key_terms = self.find_in_tokens(text.tokens)
        if not spans:
            return []
        matches = [
            KeyTermMatch(start=span.start, stop=span.stop, src=text, key_term=key_term)
            for span, key_term in zip(spans, key_terms)
        ]
        return _remove_duplicate_matches(matches) if allow_subset_matches else _remove_subset_matches(matches)


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
