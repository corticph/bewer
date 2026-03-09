from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Union

import ahocorasick

from bewer.core.text import Text, TextType, TokenList
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.example import Example

__all__ = ["Keyword", "KeywordNotFoundWarning"]


class KeywordNotFoundWarning(UserWarning):
    pass


warnings.filterwarnings("always", category=KeywordNotFoundWarning)


class Keyword(Text):
    """A keyword that can locate itself within reference text tokens.

    Inherits standardized, tokens, and pipeline caching from Text.
    Adds contiguous token matching against a reference TokenList.
    """

    def __init__(
        self,
        raw: str,
        src: Optional[Union["Example", "Dataset"]] = None,
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

        # Collect token patterns from keywords, tracking origin
        patterns = []
        self._pattern_keywords: dict[tuple[int, ...], set[str]] = {}
        keyword_patterns = []
        for keyword in keywords:
            tokens = keyword.tokens.normalized if normalized else keyword.tokens.raw
            token_pattern = tuple(tokens)
            if not token_pattern:
                continue
            keyword_patterns.append((keyword.raw, token_pattern))
            patterns.append(token_pattern)

        # Handle capitalization variants
        if add_capitalized and not normalized:
            for _, p in keyword_patterns:
                first_cap = p[0].capitalize()
                if first_cap != p[0]:
                    patterns.append((first_cap,) + p[1:])

        # Build vocab: token string -> int for KEY_SEQUENCE mode
        self._vocab = {w: i for i, w in enumerate({w for p in patterns for w in p})}
        self._unknown = len(self._vocab)

        # Map int_pattern -> set of keyword raw strings (for warn_missing)
        for kw_raw, token_pattern in keyword_patterns:
            int_pattern = tuple(self._vocab[w] for w in token_pattern)
            self._pattern_keywords.setdefault(int_pattern, set()).add(kw_raw)

        # Build Aho-Corasick automaton
        self._automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_SEQUENCE)
        seen = set()
        for pattern in patterns:
            int_pattern = tuple(self._vocab[w] for w in pattern)
            if int_pattern not in seen:
                self._automaton.add_word(int_pattern, len(pattern))
                seen.add(int_pattern)
        self._automaton.make_automaton()

    def find_in_tokens(
        self,
        tokens: TokenList,
        allow_subsets: bool = True,
        warn_missing: bool = False,
    ) -> list[slice]:
        """Find all contiguous token sequences that match any keyword in the automaton.

        Args:
            tokens: The token list to search in.
            allow_subsets: Whether to allow subset matches (default True).
            warn_missing: If True, emit a KeywordNotFoundWarning for each keyword whose
                token pattern was not found in the text. Uses tokens.src to identify the example.
        """
        token_strings = tokens.normalized if self.normalized else tokens.raw
        int_text = tuple(self._vocab.get(w, self._unknown) for w in token_strings)

        matches = []
        found_keywords: set[str] | None = set() if warn_missing else None
        for end_idx, pattern_len in self._automaton.iter(int_text):
            start_idx = end_idx - pattern_len + 1
            matches.append(slice(start_idx, end_idx + 1))
            if found_keywords is not None:
                int_pattern = int_text[start_idx : end_idx + 1]
                for kw_raw in self._pattern_keywords.get(int_pattern, ()):
                    found_keywords.add(kw_raw)

        if warn_missing:
            example_index = getattr(getattr(tokens.src, "src", None), "index", None)
            for kw_raws in self._pattern_keywords.values():
                for kw_raw in kw_raws:
                    if kw_raw not in found_keywords:
                        warnings.warn(
                            f"Keyword '{kw_raw}' not found in reference tokens: Example {example_index}.",
                            KeywordNotFoundWarning,
                        )

        if not allow_subsets:
            matches = _remove_subset_matches(matches)

        return matches


def _remove_duplicate_matches(matches: list[slice]) -> list[slice]:
    """Remove exact duplicate matches, preserving order."""
    seen: set[tuple[int, int]] = set()
    result = []
    for m in matches:
        key = (m.start, m.stop)
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


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


def get_keyword_trie(
    vocabs: dict[str, set[Keyword]],
    cache: dict[tuple, Optional[KeywordTrie]],
    vocab: str,
    normalized: bool = True,
    add_capitalized: bool = False,
) -> Optional[KeywordTrie]:
    """Get or build a trie for the keywords in the specified vocabulary."""
    trie_key = (
        STANDARDIZER_NAME.get(),
        TOKENIZER_NAME.get(),
        NORMALIZER_NAME.get() if normalized else None,
        add_capitalized,
        vocab,
    )
    if trie_key in cache:
        return cache[trie_key]

    keywords = vocabs.get(vocab, None)
    if keywords is None:
        return None

    trie = KeywordTrie(
        keywords,
        normalized=normalized,
        add_capitalized=add_capitalized,
    )
    cache[trie_key] = trie
    return trie
