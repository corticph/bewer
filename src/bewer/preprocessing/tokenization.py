"""Module for tokenization using regex patterns.

Functions returning regex patterns are generally advised to return uncompiled patterns to allow combination, but
compiled patterns are also allowed, which may be necessary when passing flags to the regex compiler.
"""

from typing import TYPE_CHECKING, Union

import regex as re

from bewer.core.text import TokenList
from bewer.core.token import Token

if TYPE_CHECKING:
    from bewer.core.text import Text


def whitespace() -> str:
    """Return a regex pattern that matches non-whitespace sequences.

    Returns:
        str: The regex pattern. Not compiled.
    """
    return r"\S+"


def whitespace_strip_symbols_and_custom(split_on: str | None = None) -> str:
    """Return a regex pattern that matches tokens without internal whitespace or punctuation per specified characters.

    Args:
        split_on (str): A string of characters to split on in addition to whitespace. Will be escaped.

    Returns:
        str: The regex pattern. Not compiled.
    """
    if split_on is not None:
        split_on = re.escape(split_on)
    letters_digits = r"\p{L}\p{N}"
    symbols_punctuation_marks = r"\p{S}\p{P}\p{M}"
    if split_on is None:
        return re.compile(rf"[{letters_digits}]+([[{symbols_punctuation_marks}]]+[{letters_digits}]+)*", re.V1)
    return re.compile(
        rf"[{letters_digits}]+([[{symbols_punctuation_marks}]--[{split_on}]]+[{letters_digits}]+)*", re.V1
    )


class Tokenizer(object):
    def __init__(
        self,
        pattern: Union[str, re.Pattern],
        name: str = "__no_name__",
    ):
        """Initialize a tokenizer with a regex pattern.

        Args:
            name (str): The config name of the tokenizer.
            pattern (Union[str, re.Pattern]): The regex pattern to use for tokenization.
        """
        self._pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self._name = name

    def __call__(self, text: str, _src_text: Union["Text", None] = None) -> TokenList:
        """Tokenize the input text using the regex pattern.

        Args:
            text (str): The input text to tokenize.
            _src_text (Text | None): The source Text object, if available.

        Returns:
            TokenList: A list of Token objects.
        """

        def from_match(match_tuple: tuple[int, re.Match]) -> Token:
            """Create a Token object from a regex match object."""
            token = Token.from_match(match_tuple[1], index=match_tuple[0], _src_text=_src_text)
            return token

        return TokenList(map(from_match, enumerate(self._pattern.finditer(text))))

    def __repr__(self):
        """Return a string representation of the Tokenizer object."""
        if self._name is None:
            return f"Tokenizer({self._pattern.pattern})"
        return f"Tokenizer({self._name}: {self._pattern.pattern})"
