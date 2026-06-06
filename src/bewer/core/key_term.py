from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import ahocorasick

from bewer.core.text import Text, TextType, TokenList

if TYPE_CHECKING:
    from bewer.core.example import Example
    from bewer.core.vocabulary import Vocabulary

__all__ = ["KeyTerm", "Match"]


class KeyTerm(Text):
    """A key term that can locate itself within reference text tokens.

    Inherits standardized, tokens, and pipeline caching from Text. A ``KeyTerm`` is
    canonical: there is one instance per raw string within a vocabulary, deduped by the
    owning :class:`Vocabulary`. Its ``src`` is that vocabulary (which back-references the
    dataset and provides the active ``pipelines``), and ``examples`` records every example
    that regards the term.
    """

    def __init__(
        self,
        raw: str,
        *,
        src: Vocabulary,
    ):
        # KeyTerm reuses Text's machinery but is sourced from a Vocabulary; Text only reads
        # ``src.pipelines``, so cast to satisfy the Example-typed base signature.
        super().__init__(raw=raw, src=cast("Example", src), text_type=TextType.KEY_TERM)
        # Examples that regard this term. Mutable back-reference; intentionally *not* part
        # of __hash__ (which stays (raw, text_type)), so canonical dedup-by-raw is unaffected.
        self.examples: set[Example] = set()

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'KeyTerm("{text}")'


@dataclass(frozen=True)
class Match:
    """A located key term occurrence within a text.

    Attributes:
        span: The token span (slice) of the match within ``text``'s tokens.
        text: The Text the match was found in.
        key_terms: The key term(s) whose token pattern produced this match. A span maps to
            more than one key term when distinct raw strings normalize to the same pattern.
    """

    span: slice
    text: Text
    key_terms: frozenset[KeyTerm]


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

        # Map each token pattern (string tuple) to the key term(s) that produce it, so a
        # matched span can be mapped back to its KeyTerm(s). Capitalized variants map back
        # to the same KeyTerm.
        pattern_to_terms: dict[tuple[str, ...], set[KeyTerm]] = {}
        for key_term in key_terms:
            tokens = key_term.tokens.normalized if normalized else key_term.tokens.raw
            token_pattern = tuple(tokens)
            if not token_pattern:
                continue
            pattern_to_terms.setdefault(token_pattern, set()).add(key_term)
            if add_capitalized and not normalized:
                first_cap = token_pattern[0].capitalize()
                if first_cap != token_pattern[0]:
                    cap_pattern = (first_cap,) + token_pattern[1:]
                    pattern_to_terms.setdefault(cap_pattern, set()).add(key_term)

        # Build vocab: token string -> int for KEY_SEQUENCE mode
        self._vocab = {w: i for i, w in enumerate({w for p in pattern_to_terms for w in p})}
        self._unknown = len(self._vocab)

        # Encoded pattern -> key term(s) reverse map.
        self._pattern_to_terms: dict[tuple[int, ...], set[KeyTerm]] = {}
        for pattern, terms in pattern_to_terms.items():
            encoded = tuple(self._vocab[w] for w in pattern)
            self._pattern_to_terms.setdefault(encoded, set()).update(terms)

        # Build Aho-Corasick automaton
        self._automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_SEQUENCE)
        for encoded in self._pattern_to_terms:
            self._automaton.add_word(encoded, len(encoded))
        self._automaton.make_automaton()

    def encode(self, tokens: TokenList) -> tuple[int, ...]:
        """Encode a token list into the trie's integer vocabulary."""
        token_strings = tokens.normalized if self.normalized else tokens.raw
        return tuple(self._vocab.get(w, self._unknown) for w in token_strings)

    def terms_for_pattern(self, pattern: tuple[int, ...]) -> set[KeyTerm]:
        """Return the key term(s) whose token pattern matches the encoded ``pattern``."""
        return self._pattern_to_terms.get(pattern, set())

    def find_in_tokens(self, tokens: TokenList) -> tuple[list[slice], list[tuple[int, ...]]]:
        """Find all key term matches, returning spans and their encoded patterns."""
        int_text = self.encode(tokens)
        matches: list[slice] = []
        patterns: list[tuple[int, ...]] = []
        for end_idx, pattern_len in self._automaton.iter(int_text):
            start = end_idx - pattern_len + 1
            matches.append(slice(start, end_idx + 1))
            patterns.append(int_text[start : end_idx + 1])
        return matches, patterns


def _remove_duplicate_matches(matches: list[Match]) -> list[Match]:
    """Remove exact duplicate matches (by span), preserving order."""
    seen: set[tuple[int, int]] = set()
    result = []
    for m in matches:
        key = (m.span.start, m.span.stop)
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _remove_subset_matches(matches: list[Match]) -> list[Match]:
    """Remove matches whose span is a subset of another match's span, preferring longer matches."""
    if not matches:
        return matches
    # Sort by start ascending, then by length descending
    ordered = sorted(matches, key=lambda m: (m.span.start, m.span.start - m.span.stop))
    result = [ordered[0]]
    for m in ordered[1:]:
        prev = result[-1]
        # Skip if fully contained within previous match
        if m.span.start >= prev.span.start and m.span.stop <= prev.span.stop:
            continue
        result.append(m)
    return result
