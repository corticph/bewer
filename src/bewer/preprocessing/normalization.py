from __future__ import annotations

import inspect
import unicodedata
from functools import lru_cache
from typing import Any, Callable

import regex as re
from unidecode import unidecode

from bewer.preprocessing._unicode_normalization_tables import APOSTROPHE_TABLE, HYPHEN_TABLE, SLASH_TABLE

__all__ = [
    "Transform",
    "TransformPipeline",
    "Normalizer",
    "Standardizer",
    "lowercase",
    "nfc",
    "normalize_apostrophe_variants",
    "normalize_hyphen_variants",
    "normalize_slash_variants",
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
# Transform — a callable with bound params
# ============================================================


def _validate_params(fn: Callable, params: dict[str, Any], *, skip_first: bool = False) -> None:
    """Validate config params against a function's signature."""
    sig = inspect.signature(fn)
    func_params = sig.parameters
    param_iter = iter(func_params.items())

    if skip_first:
        first_param = next(param_iter)[0]
        if first_param in params:
            raise ValueError(f"First positional argument '{first_param}' should not be passed in params")

    for param, value in param_iter:
        if value.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if value.default is inspect.Parameter.empty and param not in params:
            raise ValueError(f"Parameter '{param}' not found in function '{fn.__name__}'")

    for param in params:
        if param not in func_params:
            raise ValueError(f"Unexpected parameter '{param}' for function '{fn.__name__}'")


class Transform:
    """A transform function bound with parameters.

    Calls ``fn(text, **params)`` when invoked.
    """

    __slots__ = ("fn", "params")

    def __init__(self, fn: Callable[..., Any], **params: Any):
        self.fn = fn
        self.params = params
        _validate_params(fn, params, skip_first=True)

    def __call__(self, text: str) -> str:
        return self.fn(text, **self.params)

    def __repr__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
            return f"Transform({self.fn.__name__}, {param_str})"
        return f"Transform({self.fn.__name__})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return self.fn is other.fn and self.params == other.params

    def __hash__(self) -> int:
        return hash((self.fn, tuple(sorted(self.params.items()))))


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


@_set_attrs(token_only=False, length_preserving=True)
def normalize_apostrophe_variants(text: str) -> str:
    """Normalize Unicode apostrophe variants (e.g. \u2019, \u02bc) to ASCII apostrophe (').

    Args:
        text (str): Input string.

    Returns:
        str: Text with apostrophe variants replaced by ASCII apostrophe.
    """
    return text.translate(APOSTROPHE_TABLE)


@_set_attrs(token_only=False, length_preserving=True)
def normalize_hyphen_variants(text: str) -> str:
    """Normalize Unicode hyphen/dash variants (e.g. \u2013, \u2014) to ASCII hyphen-minus (-).

    Args:
        text (str): Input string.

    Returns:
        str: Text with hyphen/dash variants replaced by ASCII hyphen-minus.
    """
    return text.translate(HYPHEN_TABLE)


@_set_attrs(token_only=False, length_preserving=True)
def normalize_slash_variants(text: str) -> str:
    """Normalize Unicode slash variants (e.g. \uff0f) to ASCII solidus (/).

    Args:
        text (str): Input string.

    Returns:
        str: Text with slash variants replaced by ASCII solidus.
    """
    return text.translate(SLASH_TABLE)


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
def _transliterate_latin_letters(char: str, preserve: str = "") -> str:
    if len(char) != 1:
        raise ValueError("Input must be a single character.")

    if char in preserve:
        return char

    if unicodedata.category(char)[0] != "L":
        return char  # Non-letter characters are returned unchanged

    if not unicodedata.name(char).startswith("LATIN"):
        return char  # Non-Latin characters are returned unchanged

    return unidecode(char)


@_set_attrs(token_only=False, length_preserving=False)
def transliterate_latin_letters(text: str, preserve: str = "") -> str:
    """Transliterate Latin Unicode letters to their ASCII equivalents.

    In most cases, this is equivalent to decomposing unicode letters with diacritical marks and
    removing them. Characters listed in `preserve` are passed through unchanged, allowing
    language-specific characters (e.g. Danish æ/ø/å) to be retained.

    Args:
        text (str): Input string.
        preserve (str): Characters to leave untouched. Default empty string preserves nothing.

    Returns:
        str: ASCII transliteration of the input, with preserved characters left intact.
    """
    return "".join(_transliterate_latin_letters(c, preserve) for c in text)


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


class TransformPipeline:
    """A pipeline of transforms applied to text or tokens.

    The pipeline holds a name (for repr and introspection) and a sequence of
    ``Transform`` objects applied in order.
    """

    def __init__(self, name: str, *transforms: Transform):
        self._name = name
        self._transforms = transforms

    @property
    def name(self) -> str:
        return self._name

    @property
    def transforms(self) -> tuple[Transform, ...]:
        return self._transforms

    def __call__(self, text: str) -> str:
        return self._apply(text)

    @lru_cache(maxsize=32768)
    def _apply(self, text: str) -> str:
        for transform in self._transforms:
            text = transform(text)
        return text

    def __repr__(self) -> str:
        func_names = " -> ".join(t.fn.__name__ for t in self._transforms)
        return f"{type(self).__name__}({self._name}: {func_names})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._name == other._name and self._transforms == other._transforms

    def __hash__(self) -> int:
        return hash((self._name, self._transforms))


class Normalizer(TransformPipeline):
    """A pipeline of transforms applied at the normalization stage."""


class Standardizer(TransformPipeline):
    """A pipeline of transforms applied at the standardization stage."""
