from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import ahocorasick

from bewer.core.text import Text, TextType, TokenList

if TYPE_CHECKING:
    from bewer.core.example import Example

__all__ = ["Keyword"]


class Keyword(Text):
    """A keyword that can locate itself within reference text tokens.

    Inherits standardized, tokens, and pipeline caching from Text.
    Adds contiguous token matching against a reference TokenList.
    """

    def __init__(
        self,
        raw: str,
        src: Optional["Example"] = None,
    ):
        super().__init__(raw=raw, src=src, text_type=TextType.KEYWORD)

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'Keyword("{text}")'


class KeywordTrie:
    """Aho-Corasick automaton for efficient multi-keyword matching in token sequences."""

    def __init__(
        self,
        keywords: set[Keyword],
        normalized: bool = True,
        add_capitalized: bool = False,
    ):
        """Initialize the automaton with the given set of keywords.

        Args:
            keywords: A set of Keyword objects to build the automaton from.
            normalized: Whether to use normalized text for matching. If False, uses raw text.
            add_capitalized: Whether to add capitalized versions of keywords for case-insensitive matching.
                Only applies if normalized is False and is only applied to the first word in an n-gram keyword.
        """
        self.normalized = normalized
        self.add_capitalized = add_capitalized

        # Collect token patterns from keywords
        patterns = []
        for keyword in keywords:
            tokens = keyword.tokens.normalized if normalized else keyword.tokens.raw
            patterns.append(tuple(tokens))

        # Handle capitalization variants
        if add_capitalized and not normalized:
            extra = []
            for p in patterns:
                first_cap = p[0].capitalize()
                if first_cap != p[0]:
                    extra.append((first_cap,) + p[1:])
            patterns.extend(extra)

        # Build vocab: token string -> int for KEY_SEQUENCE mode
        self._vocab = {w: i for i, w in enumerate({w for p in patterns for w in p})}
        self._unknown = len(self._vocab)

        # Build Aho-Corasick automaton
        self._automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_SEQUENCE)
        for pattern in patterns:
            int_pattern = tuple(self._vocab[w] for w in pattern)
            self._automaton.add_word(int_pattern, len(pattern))
        self._automaton.make_automaton()

    def find_in_tokens(self, tokens: TokenList, allow_subsets: bool = True) -> list[slice]:
        """Find all contiguous token sequences that match any keyword in the automaton."""
        token_strings = tokens.normalized if self.normalized else tokens.raw
        vocab = self._vocab
        unknown = self._unknown
        int_text = tuple(vocab.get(w, unknown) for w in token_strings)

        matches = []
        for end_idx, pattern_len in self._automaton.iter(int_text):
            start_idx = end_idx - pattern_len + 1
            matches.append(slice(start_idx, end_idx + 1))

        if not allow_subsets:
            matches = _remove_subset_matches(matches)

        return matches


def _remove_subset_matches(matches: list[slice]) -> list[slice]:
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
