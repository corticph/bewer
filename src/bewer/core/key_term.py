from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Union

import ahocorasick

from bewer.core.text import Text, TextType, TokenList
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.example import Example

__all__ = [
    "KeyTerm",
    "KeyTermNotFoundWarning",
    "Vocabulary",
    "EnumeratedVocabulary",
    "FunctionVocabulary",
    "VocabularyFunction",
]

# A function-based vocabulary scans a text's tokens and returns the token-index spans it matches.
# It receives the full TokenList (exposing both ``.raw`` and ``.normalized`` token strings) and the
# ``normalized`` flag of the current matching request, so it can match the same surface form the
# caller asked for. Returned spans may cover one or several tokens.
VocabularyFunction = Callable[["TokenList", bool], "list[slice]"]


class KeyTermNotFoundWarning(UserWarning):
    pass


warnings.filterwarnings("always", category=KeyTermNotFoundWarning)


class KeyTerm(Text):
    """A key term that can locate itself within reference text tokens.

    Inherits standardized, tokens, and pipeline caching from Text.
    Adds contiguous token matching against a reference TokenList.
    """

    def __init__(
        self,
        raw: str,
        src: Optional[Union["Example", "Dataset"]] = None,
    ):
        super().__init__(raw=raw, src=src, text_type=TextType.KEY_TERM)

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'KeyTerm("{text}")'


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

        patterns = []
        key_term_patterns = []
        for key_term in key_terms:
            tokens = key_term.tokens.normalized if normalized else key_term.tokens.raw
            token_pattern = tuple(tokens)
            if not token_pattern:
                continue
            key_term_patterns.append((key_term.raw, token_pattern))
            patterns.append(token_pattern)

        # Handle capitalization variants.
        # Gated to raw matching: this assumes that when normalized=True the normalizer has already
        # folded case (true for normalizers.default, which lowercases). That assumption does NOT
        # hold for a normalizer without lowercasing (e.g. normalizers.legacy = NFC only) — there,
        # normalized tokens keep their original case yet this widening is still suppressed, so
        # case-insensitive matching is unavailable in that config.
        # Note this is an approximation, not full case-insensitivity: only the first token is varied
        # (via str.capitalize), so it catches sentence-initial / proper-noun spellings but not
        # all-caps or internally-cased variants.
        if add_capitalized and not normalized:  # TODO: Maybe drop the `and not normalized` condition.
            for _, p in key_term_patterns:
                first_cap = p[0].capitalize()
                if first_cap != p[0]:
                    patterns.append((first_cap,) + p[1:])

        # Build vocab: token string -> int for KEY_SEQUENCE mode
        self._vocab = {w: i for i, w in enumerate({w for p in patterns for w in p})}
        self._unknown = len(self._vocab)

        # Build Aho-Corasick automaton
        self._automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_SEQUENCE)
        seen = set()
        for pattern in patterns:
            int_pattern = tuple(self._vocab[w] for w in pattern)
            if int_pattern not in seen:
                self._automaton.add_word(int_pattern, len(pattern))
                seen.add(int_pattern)
        self._automaton.make_automaton()

    def encode(self, tokens: TokenList) -> tuple[int, ...]:
        """Encode a token list into the trie's integer vocabulary."""
        token_strings = tokens.normalized if self.normalized else tokens.raw
        return tuple(self._vocab.get(w, self._unknown) for w in token_strings)

    def encode_variants(self, tokens: TokenList) -> set[tuple[int, ...]]:
        """Return all encoded patterns for a token list, including capitalized variant if enabled."""
        variants = {self.encode(tokens)}
        if self.add_capitalized and not self.normalized and tokens:
            raw = tokens.raw
            cap_first = raw[0].capitalize()
            if cap_first != raw[0]:
                variants.add(tuple(self._vocab.get(w, self._unknown) for w in [cap_first] + raw[1:]))
        return variants

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


def get_key_term_trie(
    vocabs: dict[str, set[KeyTerm]],
    cache: dict[tuple, Optional[KeyTermTrie]],
    vocab: str,
    normalized: bool = True,
    add_capitalized: bool = False,
) -> Optional[KeyTermTrie]:
    """Get or build a trie for the key terms in the specified vocabulary."""
    trie_key = (
        STANDARDIZER_NAME.get(),
        TOKENIZER_NAME.get(),
        NORMALIZER_NAME.get() if normalized else None,
        add_capitalized,
        vocab,
    )
    if trie_key in cache:
        return cache[trie_key]

    key_terms = vocabs.get(vocab, None)
    if not key_terms:
        cache[trie_key] = None
        return None

    trie = KeyTermTrie(
        key_terms,
        normalized=normalized,
        add_capitalized=add_capitalized,
    )
    cache[trie_key] = trie
    return trie


class Vocabulary(ABC):
    """A named vocabulary that locates key term spans within a Text's tokens.

    A vocabulary defines *which* token spans count as key terms. Subclasses differ only in how
    membership is decided: :class:`EnumeratedVocabulary` matches against an explicit set of key
    terms (via an Aho-Corasick trie), while :class:`FunctionVocabulary` defers to a user-supplied
    function. Downstream metrics (KTR, KTP, KTF, ...) consume the resulting spans without caring
    how they were produced.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def find_matches(
        self,
        text: Text,
        *,
        normalized: bool = True,
        add_capitalized: bool = False,
        only_local_matches: bool = False,
    ) -> list[slice]:
        """Return token-index spans where this vocabulary matches ``text``.

        The returned spans are *raw*: duplicate and subset cleanup is applied by the caller
        (:meth:`Text.get_key_term_matches`), so it is intentionally not a parameter here and the
        contract stays identical across vocabulary types. Not every parameter is meaningful for
        every subclass; ones that do not apply are accepted for interface uniformity and ignored
        (see each subclass for specifics).

        Args:
            text: The text whose tokens are searched. ``text.src`` gives the owning example (and
                thereby the dataset), which subclasses may use to resolve their term set.
            normalized: Match against the normalized token surface (``text.tokens.normalized``)
                rather than the raw surface. Both subclasses honor this.
            add_capitalized: Only honored by :class:`EnumeratedVocabulary`, and only when
                ``normalized=False``: also match a leading-capitalized variant of each key term.
            only_local_matches: Only honored by :class:`EnumeratedVocabulary`: restrict a globally
                pooled vocabulary to the example's own annotated key terms.

        Returns:
            A list of slices over ``text.tokens``, each covering one matched key term span. May
            contain duplicate or overlapping spans; the caller deduplicates.
        """
        raise NotImplementedError


class EnumeratedVocabulary(Vocabulary):
    """A vocabulary backed by an explicit set of key terms, matched with an Aho-Corasick trie.

    The key terms themselves live on the owning :class:`~bewer.core.dataset.Dataset` (global vocab)
    and :class:`~bewer.core.example.Example` objects (per-example local terms); this wrapper reads
    them on demand so the existing storage and caching are reused unchanged.
    """

    def __init__(self, name: str, dataset: "Dataset"):
        super().__init__(name)
        self.dataset = dataset

    def find_matches(
        self,
        text: Text,
        *,
        normalized: bool = True,
        add_capitalized: bool = False,
        only_local_matches: bool = False,
    ) -> list[slice]:
        example = text.src
        dataset = self.dataset
        vocab = self.name

        has_local = example is not None and vocab in example.key_terms
        has_global = vocab in dataset._global_key_term_vocabs

        if not has_local and not has_global:
            return []

        tokens = text.tokens
        matches: list[slice] = []

        global_trie = (
            dataset._get_key_term_trie(vocab, normalized=normalized, add_capitalized=add_capitalized)
            if has_global
            else None
        )

        if global_trie is not None:
            raw_matches, raw_patterns = global_trie.find_in_tokens(tokens)

            if only_local_matches and has_local:
                local_int_patterns: set[tuple[int, ...]] = set()
                for kt in example.key_terms[vocab]:
                    local_int_patterns.update(global_trie.encode_variants(kt.tokens))
                matches = [m for m, p in zip(raw_matches, raw_patterns) if p in local_int_patterns]
            else:
                matches = raw_matches

            if text.text_type == TextType.REF and has_local:
                matched_patterns = set(raw_patterns)
                for kt in example.key_terms[vocab]:
                    if not matched_patterns.intersection(global_trie.encode_variants(kt.tokens)):
                        warnings.warn(
                            f"Key term '{kt.raw}' not found in reference tokens: Example {example.index}.",
                            KeyTermNotFoundWarning,
                        )

        return matches


class FunctionVocabulary(Vocabulary):
    """A vocabulary whose members are defined functionally rather than enumerated.

    The wrapped function receives a :class:`~bewer.core.text.TokenList` and the ``normalized`` flag
    of the current matching request, and returns the token-index spans (``list[slice]``) it matches.
    This makes it possible to express open-ended vocabularies such as "any alphanumeric term"
    (e.g. ``MRI``, ``HbA1c``) via a regular expression, without listing every possible term in advance.

    ``add_capitalized`` and ``only_local_matches`` are accepted for interface uniformity but ignored.
    ``add_capitalized`` is a trie-specific pattern-expansion detail of :class:`EnumeratedVocabulary`,
    whereas a function controls case handling directly within its own matching logic.
    ``only_local_matches`` scopes a globally-pooled enumerated vocabulary back down to an example's
    own annotated terms; a function vocabulary has no such pool (it is evaluated fresh per text), so
    the flag is a no-op — consistent with a global-only enumerated vocabulary, which also ignores it.
    """

    def __init__(self, name: str, fn: VocabularyFunction):
        super().__init__(name)
        if not callable(fn):
            raise TypeError(f"Vocabulary function for '{name}' must be callable, got {type(fn)}.")
        self.fn = fn

    def find_matches(
        self,
        text: Text,
        *,
        normalized: bool = True,
        add_capitalized: bool = False,
        only_local_matches: bool = False,
    ) -> list[slice]:
        tokens = text.tokens
        matches = list(self.fn(tokens, normalized))

        n = len(tokens)
        for match in matches:
            if not isinstance(match, slice):
                raise TypeError(
                    f"Function vocabulary '{self.name}' must return a list of slices, got {type(match)}."
                )
            start, stop = match.start, match.stop
            if match.step not in (None, 1) or start is None or stop is None or not 0 <= start < stop <= n:
                raise ValueError(
                    f"Function vocabulary '{self.name}' returned an invalid span {match} for a token "
                    f"sequence of length {n}; spans must be contiguous slices within [0, {n})."
                )
        return matches
