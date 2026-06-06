from __future__ import annotations

from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Iterable, Mapping, Optional, Protocol, TypeVar, Union, overload

import regex as re

from bewer.core.caching import pipeline_cached_property
from bewer.core.token import Token
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.example import Example
    from bewer.core.key_term import Match

__all__ = ["HasPipelines", "Text", "TextType", "TokenizedText", "TokenList"]


class TextType(str, Enum):
    REF = "ref"
    HYP = "hyp"
    KEY_TERM = "key_term"


class HasPipelines(Protocol):
    """Protocol for objects that expose the active preprocessing ``pipelines``.

    Implemented by :class:`Example` (the source of a ref/hyp :class:`Text`) and
    :class:`Vocabulary` (the source of a :class:`KeyTerm`); it is the bound on a
    :class:`TokenizedText`'s source type.
    """

    @property
    def pipelines(self) -> Mapping[str, Any]: ...


def _join_tokens(tokens: TokenList, normalized: bool = True) -> str:
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


S = TypeVar("S", bound=HasPipelines)


class TokenizedText(Generic[S]):
    """Shared text machinery, generic over its source type ``S`` (a :class:`HasPipelines`).

    Owns the preprocessing pipeline (standardization -> tokenization), the per-pipeline
    caches, and the ``raw``/``src``/``text_type`` accessors. :class:`Text` (sourced from an
    :class:`Example`) and :class:`KeyTerm` (sourced from a :class:`Vocabulary`) are sibling
    subclasses; each binds ``S`` to its concrete source so ``src`` types precisely.

    Attributes:
        raw (str): The original text string.
        standardized (str): The standardized text string after applying the active standardizer.
        src (S): The source object that this text belongs to and resolves pipelines from.
        text_type (TextType): The type of the text (reference, hypothesis, or key term).
    """

    def __init__(
        self,
        raw: str | None,
        *,
        src: S,
        text_type: Optional[TextType] = None,
    ):
        """Initialize the TokenizedText object.

        Args:
            raw: The original text string (reference, hypothesis, or key term).
            src: The source object that resolves pipelines (required).
            text_type: The type of the text (REF, HYP, or KEY_TERM).
        """
        self._raw = raw
        self._text_type = text_type

        self._cache_standardized = {}
        self._cache_tokens = {}

        self._src = src
        self._pipelines = src.pipelines

    @property
    def raw(self) -> str:
        if self._raw is None:
            raise ValueError("Raw text is None and lazy inference is not implemented.")
        return self._raw

    @property
    def src(self) -> S:
        """Get the source object that this text belongs to."""
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
    def tokens(self, tokenizer) -> TokenList:
        """The list of Token objects produced by the active tokenizer."""
        return TokenList.from_matches(tokenizer(self.standardized), src=self)

    def joined(self, normalized: bool = True) -> str:
        """Get the joined text from tokens.

        Args:
            normalized (bool): Whether to use normalized tokens.
        """
        return _join_tokens(self.tokens, normalized=normalized)

    def __hash__(self):
        return hash((self.raw, self._text_type))

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'{type(self).__name__}("{text}")'


class Text(TokenizedText["Example"]):
    """BeWER reference/hypothesis text, sourced from an :class:`Example`.

    Adds reference/hypothesis-only behaviour (:meth:`get_key_term_matches`) on top of the
    shared :class:`TokenizedText` machinery. Its ``src`` is the owning :class:`Example`.
    """

    def get_key_term_matches(
        self,
        vocab: str,
        normalized: bool = True,
        add_capitalized: bool = False,
        allow_subset_matches: bool = False,
        only_local_matches: bool = False,
    ) -> list[Match]:
        """Find key term matches in this text's tokens.

        Matching is delegated to the named :class:`Vocabulary`, which matches the union of
        its terms here. With ``only_local_matches=True`` only terms regarded by this text's
        example are kept. When matching on the reference side, a warning is logged for each
        term the example regards that is absent from the reference tokens.

        Args:
            vocab: Vocabulary name to match against.
            normalized: Use normalized tokens for matching.
            add_capitalized: Add capitalized first-token variants (raw mode only).
            allow_subset_matches: If False, discard matches that are subsets of longer matches.
            only_local_matches: If True, scope matches to terms regarded by this text's example.

        Returns:
            List of :class:`Match` objects representing matched token spans and their key terms.
        """
        example: Example = self._src
        dataset = example.src

        vocabulary = dataset._vocabularies.get(vocab)
        if vocabulary is None:
            return []

        return vocabulary.find_in(
            self,
            normalized=normalized,
            add_capitalized=add_capitalized,
            allow_subset_matches=allow_subset_matches,
            only_local_matches=only_local_matches,
        )


class TokenList(tuple[Token, ...]):
    """An immutable sequence of Token objects."""

    def __new__(cls, iterable=(), src=None):
        return super().__new__(cls, iterable)

    def __init__(self, iterable=(), src: Optional[TokenizedText] = None):
        self._normalized_index_cache: dict[str, dict[str, set[int]]] = {}
        self._normalized_cache: dict[str, list[str]] = {}
        self._src = src

    @property
    def src(self) -> Optional[TokenizedText]:
        """Get the source TokenizedText object, if any.

        Unlike the rest of the hierarchy, a ``TokenList`` is not required to have a
        ``src``: it is pure metadata that nothing reads, and there is no single owning
        ``TokenizedText`` for the aggregate produced by ``TextTokenList.flat`` or for a
        cross-source concatenation. Individual ``Token`` objects always carry their own
        (required) ``TokenizedText`` src, which is what drives normalization/pipeline resolution.
        """
        return self._src

    @classmethod
    def from_matches(
        cls,
        matches: Iterable[re.Match],
        src: TokenizedText,
    ) -> TokenList:
        """Create a TokenList from an iterable of regex match objects.

        Args:
            matches: An iterable of regex Match objects.
            src: The source TokenizedText object (a ``Text`` or ``KeyTerm``).

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

    @overload
    def __getitem__(self, index: int) -> Token: ...

    @overload
    def __getitem__(self, index: slice) -> TokenList: ...

    def __getitem__(self, index: int | slice) -> Union[Token, TokenList]:
        if isinstance(index, slice):
            return TokenList(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: TokenList) -> TokenList:
        return TokenList(super().__add__(other))

    def __repr__(self):
        tokens = self[:60]
        tokens_str = ",\n ".join([repr(token) for token in tokens])
        if len(self) > 60:
            tokens_str += ",\n ..."
        return f"TokenList([\n {tokens_str}]\n)"
