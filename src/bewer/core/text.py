from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Optional, Union

import regex as re

from bewer.core.caching import pipeline_cached_property
from bewer.core.token import Token
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.example import Example


class TextType(str, Enum):
    REF = "ref"
    HYP = "hyp"
    KEYWORD = "keyword"


def _join_tokens(tokens: "TokenList", normalized: bool = True) -> str:
    """Join a list of tokens into a single string with a specified delimiter.

    Args:
        tokens (list[str]): The list of tokens to join.

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
        text_type (TextType): The type of the text (reference, hypothesis, or keyword).
    """

    def __init__(
        self,
        raw: str | None,
        src: Optional["Example"] = None,
        text_type: Optional[TextType] = None,
    ):
        """Initialize the Text object.

        Args:
            raw: The original text string (reference, hypothesis, or keyword).
            src: Parent Example object. Can be set later via set_source().
            text_type: The type of the text (REF, HYP, or KEYWORD).
        """
        self._raw = raw
        self._text_type = text_type

        self._cache_standardized = {}
        self._cache_tokens = {}

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

    @pipeline_cached_property(STANDARDIZER_NAME)
    def standardized(self, standardizer):
        """The standardized text string after applying the active standardizer."""
        return standardizer(self.raw)

    @pipeline_cached_property(TOKENIZER_NAME)
    def tokens(self, tokenizer):
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
        if src is not None and src.src is not None:
            self._pipelines = src.src.pipelines
        else:
            self._pipelines = None

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
        return f'Text("{text}")'


class TokenList(list["Token"]):
    """A list of Token objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._normalized_index_cache: dict[str, dict[str, list[int]]] = {}

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
        return cls(Token.from_match(match, index=i, src=src) for i, match in enumerate(matches))

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
        return [token.normalized for token in self]

    @cached_property
    def _start_index_mapping(self) -> dict[int, int]:
        """Create a mapping from character start index to token index for quick lookup."""
        mapping = {}
        for i, token in enumerate(self):
            mapping[token.start] = i
        return mapping

    @cached_property
    def _end_index_mapping(self) -> dict[int, int]:
        """Create a mapping from character end index to token index for quick lookup."""
        mapping = {}
        for i, token in enumerate(self):
            mapping[token.end] = i
        return mapping

    def start_index_to_token(self, char_index: int) -> Optional["Token"]:
        """Get the token that starts at the given character index.

        Args:
            char_index (int): The character index to look up.

        Returns:
            Optional[Token]: The token that starts at the given character index, or None if not found.
        """
        token_index = self._start_index_mapping.get(char_index, None)
        if token_index is not None:
            return self[token_index]
        return None

    def end_index_to_token(self, char_index: int) -> Optional["Token"]:
        """Get the token that ends at the given character index.

        Args:
            char_index (int): The character index to look up.

        Returns:
            Optional[Token]: The token that ends at the given character index, or None if not found.
        """
        token_index = self._end_index_mapping.get(char_index, None)
        if token_index is not None:
            return self[token_index]
        return None

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

    def __getitem__(self, index: int) -> Union["Token", "TokenList"]:
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
