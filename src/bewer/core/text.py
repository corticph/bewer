import warnings
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Optional, Union

import regex as re

from bewer.core.caching import pipeline_cached_property
from bewer.core.token import Token
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.example import Example

__all__ = ["Text", "TextType", "TokenList"]


class TextType(str, Enum):
    REF = "ref"
    HYP = "hyp"
    KEY_TERM = "key_term"


def _join_tokens(tokens: "TokenList", normalized: bool = True) -> str:
    """Join tokens into a single string, preserving original spacing.

    Args:
        tokens (TokenList): The tokens to join.
        normalized (bool): Whether to use normalized token text.

    Returns:
        str: The joined string.
    """
    joined = ""
    prev_end = 0
    for token in tokens:
        if token.start > prev_end:
            joined += f" {token.normalized}" if normalized else f" {token.raw}"
        else:
            joined += token.normalized if normalized else token.raw
        prev_end = token.end
    return joined.strip()


class Text:
    """BeWER text representation.

    Attributes:
        raw (str): The original text string.
        standardized (str): The standardized text string after applying the specified standardizer.
        src (Example): The source Example object that this text belongs to.
        text_type (TextType): The type of the text (reference, hypothesis, or key term).
    """

    def __init__(
        self,
        raw: str | None,
        src: Optional["Example"] = None,
        text_type: Optional[TextType] = None,
    ):
        """Initialize the Text object.

        Args:
            raw: The original text string (reference, hypothesis, or key term).
            src: Parent Example object. Can be set later via set_source().
            text_type: The type of the text (REF, HYP, or KEY_TERM).
        """
        self._raw = raw
        self._text_type = text_type

        self._cache_standardized = {}
        self._cache_tokens = {}
        self._cache_key_term_matches = {}

        self._src = None
        self._pipelines = None
        if src is not None:
            self.set_source(src)

    @property
    def raw(self) -> str:
        if self._raw is None:
            raise ValueError("Raw text is None and lazy inference is not implemented.")
        return self._raw

    @property
    def src(self) -> Optional["Example"]:
        """Get the parent Example object."""
        return self._src

    @property
    def text_type(self) -> Optional[TextType]:
        return self._text_type

    @property
    def pipelines(self):
        return self._pipelines

    @pipeline_cached_property(STANDARDIZER_NAME)
    def standardized(self, standardizer) -> str:
        """The standardized text string after applying the active standardizer."""
        return standardizer(self.raw)

    @pipeline_cached_property(TOKENIZER_NAME)
    def tokens(self, tokenizer) -> "TokenList":
        """The list of Token objects produced by the active tokenizer."""
        return TokenList.from_matches(tokenizer(self.standardized), src=self)

    def set_source(self, src: "Example") -> None:
        """Set the parent Example object.

        Args:
            src: The parent Example object.

        Raises:
            ValueError: If source is already set.
        """
        if self._src is not None:
            raise ValueError("Source already set for Text")

        self._src = src

        # Cache pipeline reference
        self._pipelines = src.pipelines if src is not None else None

    def joined(self, normalized: bool = True) -> str:
        """Get the joined text from tokens.

        Args:
            normalized (bool): Whether to use normalized tokens.
        """
        return _join_tokens(self.tokens, normalized=normalized)

    def get_key_term_matches(
        self,
        vocab: str,
        normalized: bool = True,
        add_capitalized: bool = False,
        allow_subset_matches: bool = True,
        only_local_matches: bool = False,
    ) -> list[slice]:
        """Find key term matches in this text's tokens.

        By default, matches against the dataset-wide global vocabulary. With
        ``only_local_matches=True``, only the per-example local key terms are used.
        When matching on the reference side, each local key term is also verified:
        a ``KeyTermNotFoundWarning`` is emitted for each local term absent from the reference tokens.

        Args:
            vocab: Vocabulary name to match against.
            normalized: Use normalized tokens for matching.
            add_capitalized: Add capitalized first-token variants (raw mode only).
            allow_subset_matches: If False, discard matches that are subsets of longer matches.
            only_local_matches: Use only per-example local key terms instead of the global vocab.

        Returns:
            List of slices representing matched token spans.
        """
        example = self._src
        dataset = example.src if example is not None else None

        has_local = example is not None and vocab in example.key_terms
        has_global = dataset is not None and vocab in dataset._global_key_term_vocabs

        if not has_local and not has_global:
            return []

        cache_key = (
            STANDARDIZER_NAME.get(),
            TOKENIZER_NAME.get(),
            NORMALIZER_NAME.get() if normalized else None,
            add_capitalized,
            allow_subset_matches,
            vocab,
            only_local_matches,
        )
        if cache_key in self._cache_key_term_matches:
            return self._cache_key_term_matches[cache_key]

        from bewer.core.key_term import (  # lazy import to avoid circular dependency
            KeyTermNotFoundWarning,
            _remove_duplicate_matches,
            _remove_subset_matches,
        )

        tokens = self.tokens
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

            if self._text_type == TextType.REF and has_local:
                matched_patterns = set(raw_patterns)
                for kt in example.key_terms[vocab]:
                    if not matched_patterns.intersection(global_trie.encode_variants(kt.tokens)):
                        warnings.warn(
                            f"Key term '{kt.raw}' not found in reference tokens: Example {example.index}.",
                            KeyTermNotFoundWarning,
                        )

        if matches:
            if allow_subset_matches:
                matches = _remove_duplicate_matches(matches)
            else:
                matches = _remove_subset_matches(matches)

        self._cache_key_term_matches[cache_key] = matches
        return matches

    def __hash__(self):
        return hash((self.raw, self._text_type))

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'Text("{text}")'


class TokenList(tuple["Token", ...]):
    """An immutable sequence of Token objects."""

    def __new__(cls, iterable=(), src=None):
        return super().__new__(cls, iterable)

    def __init__(self, iterable=(), src: Optional["Text"] = None):
        self._normalized_index_cache: dict[str, dict[str, set[int]]] = {}
        self._normalized_cache: dict[str, list[str]] = {}
        self._src = src

    @property
    def src(self) -> Optional["Text"]:
        """Get the source Text object."""
        return self._src

    @classmethod
    def from_matches(
        cls,
        matches: "Iterable[re.Match]",
        src: Optional["Text"] = None,
    ) -> "TokenList":
        """Create a TokenList from an iterable of regex match objects.

        Args:
            matches: An iterable of regex Match objects.
            src: The source Text object, if available.

        Returns:
            TokenList: A list of Token objects created from the matches.
        """
        return cls((Token.from_match(match, index=i, src=src) for i, match in enumerate(matches)), src=src)

    @property
    def raw(self) -> list[str]:
        """Get the raw tokens as a regular Python list.

        Returns:
            list[str]: The raw tokens.
        """
        return [token.raw for token in self]

    @property
    def normalized(self) -> list[str]:
        """Get the normalized tokens as a regular Python list.

        Returns:
            list[str]: The normalized tokens.
        """
        key = NORMALIZER_NAME.get()
        if key not in self._normalized_cache:
            self._normalized_cache[key] = [token.normalized for token in self]
        return self._normalized_cache[key]

    def ngrams(
        self,
        n: int,
        normalized: bool = True,
        join_tokens: bool = True,
    ) -> list[str]:
        """Get n-grams from the token list.

        Args:
            n (int): The size of the n-grams.
            normalized (bool): Whether to use normalized tokens.
            join_tokens (bool): Whether to join tokens into a single string.

        Returns:
            list[str]: The list of n-grams.
        """
        if n < 1:
            raise ValueError("n must be a positive integer")
        if n == 1:
            return self.raw if not normalized else self.normalized
        ngrams = []
        for i in range(len(self) - n + 1):
            ngram = self[i : i + n]
            if join_tokens:
                ngram = _join_tokens(ngram, normalized=normalized)
            else:
                ngram = ngram.normalized if normalized else ngram.raw
            ngrams.append(ngram)
        return ngrams

    @cached_property
    def _raw_index_mapping(self) -> dict[str, set[int]]:
        """Mapping from raw token text to set of positions in this TokenList."""
        mapping: dict[str, set[int]] = {}
        for i, token in enumerate(self):
            mapping.setdefault(token.raw, set()).add(i)
        return mapping

    @property
    def _normalized_index_mapping(self) -> dict[str, set[int]]:
        """Mapping from normalized token text to set of positions, cached per normalizer."""
        key = NORMALIZER_NAME.get()
        cache = self._normalized_index_cache
        if key not in cache:
            mapping: dict[str, set[int]] = {}
            for i, token in enumerate(self):
                mapping.setdefault(token.normalized, set()).add(i)
            cache[key] = mapping
        return cache[key]

    def indices(self, text: str, normalized: bool = True) -> set[int]:
        """Find all token positions where the token text matches the given string.

        Args:
            text: The string to search for.
            normalized: If True, compare against normalized token text.
                        If False, compare against raw token text.

        Returns:
            set[int]: Set of indices where the token's text matches.
        """
        mapping = self._normalized_index_mapping if normalized else self._raw_index_mapping
        return mapping.get(text, set())

    def _sub_repr(self):
        """Used internally by TextTokenList.__repr__"""
        tokens = self[:5]
        tokens_str = ",  ".join([repr(token) for token in tokens])
        if len(self) > 5:
            tokens_str += ", ..."
        return f"TokenList([{tokens_str}])"

    def __getitem__(self, index: int | slice) -> Union["Token", "TokenList"]:
        if isinstance(index, slice):
            return TokenList(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: "TokenList") -> "TokenList":
        return TokenList(super().__add__(other))

    def __repr__(self):
        tokens = self[:60]
        tokens_str = ",\n ".join([repr(token) for token in tokens])
        if len(self) > 60:
            tokens_str += ",\n ..."
        return f"TokenList([\n {tokens_str}]\n)"
