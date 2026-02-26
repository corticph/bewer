import unicodedata
from functools import lru_cache

import regex as re
from unidecode import unidecode

__all__ = [
    "Normalizer",
    "lowercase",
    "nfc",
    "strip_punctuation",
    "transliterate_latin_letters",
    "transliterate_symbols",
    "remove_symbols",
]

# ============================================================
# Helpers
# ============================================================


_VALID_ATTRS = {
    "token_only": bool,
    "length_preserving": bool,
}


def _set_attrs(**attrs):
    def decorator(func):
        for k, v in attrs.items():
            # if k not in _VALID_ATTRS:
            #     raise ValueError(f"Invalid attribute: {k}")
            # if not isinstance(v, _VALID_ATTRS[k]):
            #     raise TypeError(f"Attribute {k} must be of type {_VALID_ATTRS[k].__name__}")
            setattr(func, k, v)
        return func

    return decorator


# ============================================================
# General transformations
# ============================================================


@_set_attrs(token_only=False, length_preserving=True)
def lowercase(text: str) -> str:
    """
    Lowercase the input string.

    Args:
        text (str): Input string.

    Returns:
        str: Lowercased string.
    """
    return text.lower()


@_set_attrs(token_only=False, length_preserving=False)
def nfc(text: str) -> str:
    """
    Normalize the input string to Unicode NFC (Normalization Form Composition).

    Args:
        text (str): Input string.

    Returns:
        str: NFC-normalized string.
    """
    return unicodedata.normalize("NFC", text)


# ============================================================
# Token-level transformations
# ============================================================


@_set_attrs(token_only=True, length_preserving=False)
def strip_punctuation(text: str) -> str:
    """
    Strip leading and trailing Unicode punctuation from the input string, excluding '&' and '%'.

    Args:
        text (str): Input string.

    Returns:
        str: Stripped string.
    """
    return re.sub(r"(?V1)^[\p{P}--[&%]]+|[\p{P}--[&%]]+$", "", text)


# ============================================================
# Character-level transformations
# ============================================================

# TODO: Handle special cases, such as & and %, which are punctutation, but should never be stripped/removed.


@lru_cache(maxsize=2048)
def _transliterate_latin_letters(char: str) -> str:
    if len(char) != 1:
        raise ValueError("Input must be a single character.")

    if unicodedata.category(char)[0] != "L":
        return char  # Non-letter characters are returned unchanged

    if not unicodedata.name(char).startswith("LATIN"):
        return char  # Non-Latin characters are returned unchanged

    return unidecode(char)


@_set_attrs(token_only=False, length_preserving=False)
def transliterate_latin_letters(text: str) -> str:
    """Transliterate a single latin Unicode letter to its ASCII equivalent.

    In most cases, this means is equivalent to decomposing unicode letters with diacritical marks and removing them.

    Args:
        text (str): Input string.

    Returns:
        str: ASCII transliteration of the character.
    """
    return "".join(_transliterate_latin_letters(c) for c in text)


@lru_cache(maxsize=2048)
def _transliterate_symbols(char: str) -> str:
    if len(char) != 1:
        raise ValueError("Input must be a single character.")

    if unicodedata.category(char)[0] in {"M", "S", "P"}:
        ascii_char = unidecode(char)
        if all(unicodedata.category(c)[0] in {"M", "S", "P"} for c in ascii_char):
            return ascii_char

    return char


@_set_attrs(token_only=False, length_preserving=False)
def transliterate_symbols(text: str) -> str:
    """Transliterate Unicode markers, symbols, and punctuation characters to their ASCII equivalents.

    Only convert characters that are also transliterated to ASCII characters that are in M, S, P categories. For
    instance, currency symbols like '€' and '£' will not be transliterated.

    Args:
        text (str): Input string.

    Returns:
        str: ASCII transliteration of the character.
    """
    return "".join(_transliterate_symbols(c) for c in text)


@lru_cache(maxsize=2048)
def _remove_symbols(char: str) -> str:
    if len(char) != 1:
        raise ValueError("Input must be a single character.")

    if unicodedata.category(char)[0] in {"M", "S", "P"}:
        return ""

    return char


@_set_attrs(token_only=False, length_preserving=False)
def remove_symbols(text: str) -> str:
    """Remove Unicode markers, symbols, and punctuation characters.

    Note that this function may remove diacritical marks from decomposed letters, but this can be handled by
    normlizing to the canonical composed form: unicodedata.normalize('NFC', text).

    Args:
        text (str): Input string.

    Returns:
        str: An empty string if the character is in M, S, P categories; otherwise, the original character.
    """
    return "".join(_remove_symbols(c) for c in text)


class Normalizer(object):
    def __init__(
        self,
        pipeline: list[callable],
        name: str = "__no_name__",
    ):
        """Initialize a normalizer with a pipeline of normalization functions.

        Args:
            name (str): The config name of the normalizer.
            pipeline (list[callable]): A list of normalization functions to apply in sequence.
        """
        self._pipeline = pipeline
        self._name = name

    def __call__(self, text: str) -> str:
        """Apply the normalization pipeline to the input text.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.
        """
        for func, kwargs in self._pipeline:
            text = func(text, **kwargs)
        return text

    def __repr__(self):
        """Return a string representation of the Normalizer object."""
        func_names = " -> ".join(func.__name__ for func, _ in self._pipeline)
        if self._name is None:
            return f"Normalizer({func_names})"
        return f"Normalizer({self._name}: {func_names})"
