from enum import StrEnum
from typing import TYPE_CHECKING, Optional, Union

import regex as re

from bewer.preprocessing.context import STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.example import Example
    from bewer.core.token import Token


class TextType(StrEnum):
    REF = "ref"
    HYP = "hyp"
    KEYWORD = "keyword"


def _find_word_slices(text, term, whole_word=True, case_sensitive=False) -> list[slice]:
    """Find all slices where word appears in text."""
    if whole_word:
        pattern = r"\b" + re.escape(term) + r"\b"
    else:
        pattern = re.escape(term)
    flags = 0 if case_sensitive else re.IGNORECASE
    return [slice(m.start(), m.end()) for m in re.finditer(pattern, text, flags)]


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
        text (str): The original text. Either reference or hypothesis.
        tokens (list[Token]): A list of Token objects.
    """

    def __init__(
        self,
        raw: str | None,
        src_example: Optional["Example"] = None,
        text_type: Optional[TextType] = None,
    ):
        """Initialize the Text object.

        Args:
            raw (str): The original text. Either reference or hypothesis.
            _src (Example): The source Example object.
            _is_hyp (bool): Whether the text is a hypothesis or not.
        """
        self._raw = raw
        self._src_example = src_example
        self._text_type = text_type

        self._standardized = {}
        self._tokenized = {}

        if src_example is not None and src_example._src_dataset is not None:
            self._pipelines = src_example._src_dataset.pipelines
        else:
            self._pipelines = None

    @property
    def raw(self) -> str:
        if self._raw is None:
            raise ValueError("Raw text is None and lazy inference is not implemented.")
        return self._raw

    @property
    def src(self) -> Optional["Example"]:
        return self._src_example

    @property
    def text_type(self) -> Optional[TextType]:
        return self._text_type

    @property
    def tokens(self) -> "TokenList":
        """Get the list of Token objects using a specified tokenizer.

        Returns:
            TokenList: The list of Token objects.
        """
        standardizer_name = STANDARDIZER_NAME.get()
        tokenizer_name = TOKENIZER_NAME.get()
        pipeline_key = (standardizer_name, tokenizer_name)

        if pipeline_key in self._tokenized:
            return self._tokenized[pipeline_key]
        if self._pipelines is None:
            raise ValueError("No tokenizers found in pipelines.")
        tokenizer = self._pipelines.tokenizers.get(tokenizer_name, None)
        if tokenizer is None:
            raise ValueError(f"Tokenizer '{tokenizer_name}' not found in pipelines.")

        tokens = tokenizer(self.standardized, _src_text=self)
        self._tokenized[pipeline_key] = tokens
        return tokens

    @property
    def standardized(self) -> str:
        """Get the standardized text using a specified standardizer.

        Args:
            standardizer (str): The name of the text standardization function to apply.

        Returns:
            str: The standardized string.
        """
        standardizer_name = STANDARDIZER_NAME.get()

        if standardizer_name in self._standardized:
            return self._standardized[standardizer_name]
        if self._pipelines is None:
            raise ValueError("No standardizers found in pipelines.")
        standardizer_func = self._pipelines.standardizers.get(standardizer_name, None)
        if standardizer_func is None:
            raise ValueError(f"Text standardizer '{standardizer_name}' not found in pipelines.")

        standardized_text = standardizer_func(self.raw)
        self._standardized[standardizer_name] = standardized_text
        return standardized_text

    def joined(self, normalized: bool = True) -> str:
        """Get the joined text from tokens.

        Args:
            normalized (bool): Whether to use normalized tokens.
        """
        return _join_tokens(self.tokens, normalized=normalized)

    def get_keyword_span(self) -> list[slice]:
        """Get the span of a keyword in the text.

        Args:
            keyword (str): The keyword to find.

        Returns:
            list[slice]: The span of the keyword in the text.
        """
        if self._text_type != TextType.KEYWORD:
            raise ValueError("get_keyword_span can only be called on Text objects of type KEYWORD.")
        if self._src_example is None:
            raise ValueError("Source example is None, cannot get keyword span.")
        return _find_word_slices(
            self._src_example.ref.standardized,
            self.standardized,
            whole_word=True,
            case_sensitive=False,
        )

    def __hash__(self):
        return hash((self.raw, self._text_type))

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'Text("{text}")'


class TokenList(list["Token"]):
    """A list of Token objects."""

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
            return self.raw if normalized else self.normalized
        ngrams = []
        for i in range(len(self) - n + 1):
            ngram = self[i : i + n]
            if join_tokens:
                ngram = _join_tokens(ngram, normalized=normalized)
            else:
                ngram = ngram.normalized if normalized else ngram.raw
            ngrams.append(ngram)
        return ngrams

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
