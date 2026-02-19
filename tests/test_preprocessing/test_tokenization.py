"""Tests for bewer.preprocessing.tokenization module."""

import regex as re

from bewer.core.text import TokenList
from bewer.preprocessing.tokenization import (
    Tokenizer,
    strip_punctuation,
    strip_punctuation_keep_symbols,
    whitespace,
)


class TestWhitespace:
    """Tests for the whitespace() pattern function."""

    def test_returns_pattern(self):
        """Test that whitespace() returns a regex pattern string."""
        pattern = whitespace()
        assert isinstance(pattern, re.Pattern)

    def test_pattern_matches_words(self):
        """Test that the pattern matches non-whitespace sequences."""
        pattern = whitespace()
        matches = pattern.findall("hello world")
        assert matches == ["hello", "world"]

    def test_pattern_handles_multiple_spaces(self):
        """Test pattern with multiple consecutive spaces."""
        pattern = whitespace()
        matches = pattern.findall("hello   world")
        assert matches == ["hello", "world"]

    def test_pattern_handles_tabs_and_newlines(self):
        """Test pattern with tabs and newlines."""
        pattern = whitespace()
        matches = pattern.findall("hello\tworld\ntest")
        assert matches == ["hello", "world", "test"]

    def test_pattern_empty_string(self):
        """Test pattern with empty string."""
        pattern = whitespace()
        matches = pattern.findall("")
        assert matches == []

    def test_pattern_whitespace_only(self):
        """Test pattern with whitespace only."""
        pattern = whitespace()
        matches = pattern.findall("   \t\n   ")
        assert matches == []

    def test_pattern_includes_punctuation(self):
        """Test that punctuation is included in tokens."""
        pattern = whitespace()
        matches = pattern.findall("hello, world!")
        assert matches == ["hello,", "world!"]


class TestWhitespaceStripSymbolsAndCustom:
    """Tests for the whitespace_strip_symbols_and_custom() pattern function."""

    def test_returns_compiled_pattern(self):
        """Test that function returns a compiled pattern."""
        pattern = strip_punctuation(None)
        assert isinstance(pattern, re.Pattern)

    def test_basic_tokenization(self):
        """Test basic tokenization without custom split chars."""
        pattern = strip_punctuation(None)
        matches = [m.group() for m in pattern.finditer("hello world")]
        assert matches == ["hello", "world"]

    def test_split_on_escaped_single_char(self):
        """Test tokenization with a single escaped split character."""
        pattern = strip_punctuation(split_on_escaped="-")
        matches = [m.group() for m in pattern.finditer("hello-world")]
        assert matches == ["hello", "world"]

    def test_hyphenated_words_preserved_without_split(self):
        """Test that hyphenated words are preserved without custom split."""
        pattern = strip_punctuation()
        matches = [m.group() for m in pattern.finditer("well-known")]
        assert matches == ["well-known"]

    def test_split_on_escaped_multiple_chars(self):
        """Test with multiple escaped split characters."""
        pattern = strip_punctuation(split_on_escaped="-_")
        matches = [m.group() for m in pattern.finditer("hello-world_test")]
        assert matches == ["hello", "world", "test"]

    def test_split_on_escaped_special_regex_chars(self):
        """Test that special regex characters in split_on_escaped are escaped."""
        pattern = strip_punctuation(split_on_escaped=".")
        matches = [m.group() for m in pattern.finditer("hello.world")]
        assert matches == ["hello", "world"]

    def test_split_on_pattern(self):
        """Test splitting with a regex pattern."""
        pattern = strip_punctuation(split_on_pattern=r"\p{Sc}")
        matches = [m.group() for m in pattern.finditer("100$50")]
        assert matches == ["100", "50"]

    def test_split_on_pattern_preserves_non_matching(self):
        """Test that split_on_pattern only splits on matching characters."""
        pattern = strip_punctuation(split_on_pattern=r"\p{Sc}")
        matches = [m.group() for m in pattern.finditer("well-known")]
        assert matches == ["well-known"]

    def test_split_on_escaped_and_pattern_combined(self):
        """Test using both split_on_escaped and split_on_pattern together."""
        pattern = strip_punctuation(split_on_escaped="-", split_on_pattern=r"\p{Sc}")
        matches = [m.group() for m in pattern.finditer("hello-world$test")]
        assert matches == ["hello", "world", "test"]


class TestStripPunctuationKeepSymbols:
    """Tests for the strip_punctuation_keep_symbols() pattern function."""

    def test_returns_compiled_pattern(self):
        pattern = strip_punctuation_keep_symbols()
        assert isinstance(pattern, re.Pattern)

    def test_basic_words(self):
        pattern = strip_punctuation_keep_symbols()
        matches = [m.group() for m in pattern.finditer("hello world")]
        assert matches == ["hello", "world"]

    def test_strips_trailing_punctuation(self):
        pattern = strip_punctuation_keep_symbols()
        matches = [m.group() for m in pattern.finditer("hello, world!")]
        assert matches == ["hello", "world"]

    def test_keeps_currency_symbols(self):
        pattern = strip_punctuation_keep_symbols()
        matches = [m.group() for m in pattern.finditer("costs $100")]
        assert matches == ["costs", "$", "100"]

    def test_keeps_euro_symbol(self):
        pattern = strip_punctuation_keep_symbols()
        matches = [m.group() for m in pattern.finditer("price is €50")]
        assert matches == ["price", "is", "€", "50"]

    def test_keeps_percent_sign(self):
        pattern = strip_punctuation_keep_symbols()
        matches = [m.group() for m in pattern.finditer("95% accuracy")]
        assert matches == ["95", "%", "accuracy"]

    def test_keeps_math_symbols(self):
        pattern = strip_punctuation_keep_symbols()
        matches = [m.group() for m in pattern.finditer("a+b=c")]
        assert matches == ["a", "+", "b", "=", "c"]

    def test_hyphenated_words_preserved(self):
        pattern = strip_punctuation_keep_symbols()
        matches = [m.group() for m in pattern.finditer("well-known fact")]
        assert matches == ["well-known", "fact"]

    def test_custom_split_on(self):
        pattern = strip_punctuation_keep_symbols("-")
        matches = [m.group() for m in pattern.finditer("well-known fact")]
        assert matches == ["well", "known", "fact"]

    def test_empty_string(self):
        pattern = strip_punctuation_keep_symbols()
        matches = [m.group() for m in pattern.finditer("")]
        assert matches == []

    def test_mixed_symbols_and_punctuation(self):
        pattern = strip_punctuation_keep_symbols()
        matches = [m.group() for m in pattern.finditer("total: $99, or €89!")]
        assert matches == ["total", "$", "99", "or", "€", "89"]


class TestTokenizer:
    """Tests for the Tokenizer class."""

    def test_init_with_string_pattern(self):
        """Test initialization with string pattern."""
        tokenizer = Tokenizer(r"\S+", name="whitespace")
        assert tokenizer._name == "whitespace"
        assert isinstance(tokenizer._pattern, re.Pattern)

    def test_init_with_compiled_pattern(self):
        """Test initialization with compiled pattern."""
        pattern = re.compile(r"\S+")
        tokenizer = Tokenizer(pattern, name="whitespace")
        assert tokenizer._pattern is pattern

    def test_basic_tokenization(self):
        """Test basic tokenization."""
        tokenizer = Tokenizer(r"\S+", name="whitespace")
        tokens = tokenizer("hello world")
        assert isinstance(tokens, TokenList)
        assert len(tokens) == 2
        assert tokens[0].raw == "hello"
        assert tokens[1].raw == "world"

    def test_token_positions(self):
        """Test that tokens have correct positions."""
        tokenizer = Tokenizer(r"\S+", name="whitespace")
        tokens = tokenizer("hello world")
        assert tokens[0].start == 0
        assert tokens[0].end == 5
        assert tokens[1].start == 6
        assert tokens[1].end == 11

    def test_token_indices(self):
        """Test that tokens have correct indices."""
        tokenizer = Tokenizer(r"\S+", name="whitespace")
        tokens = tokenizer("one two three")
        assert tokens[0].index == 0
        assert tokens[1].index == 1
        assert tokens[2].index == 2

    def test_empty_string(self):
        """Test tokenization of empty string."""
        tokenizer = Tokenizer(r"\S+", name="whitespace")
        tokens = tokenizer("")
        assert len(tokens) == 0

    def test_whitespace_only(self):
        """Test tokenization of whitespace-only string."""
        tokenizer = Tokenizer(r"\S+", name="whitespace")
        tokens = tokenizer("   \t\n   ")
        assert len(tokens) == 0

    def test_repr_with_name(self):
        """Test string representation with name."""
        tokenizer = Tokenizer(r"\S+", name="test_tokenizer")
        repr_str = repr(tokenizer)
        assert "test_tokenizer" in repr_str
        assert r"\S+" in repr_str

    def test_repr_without_name(self):
        """Test string representation without name."""
        tokenizer = Tokenizer(r"\S+", name=None)
        repr_str = repr(tokenizer)
        assert r"\S+" in repr_str

    def test_leading_trailing_whitespace(self):
        """Test tokenization with leading/trailing whitespace."""
        tokenizer = Tokenizer(r"\S+", name="whitespace")
        tokens = tokenizer("  hello world  ")
        assert len(tokens) == 2
        assert tokens[0].raw == "hello"
        assert tokens[0].start == 2

    def test_punctuation_in_tokens(self):
        """Test that punctuation is included in tokens."""
        tokenizer = Tokenizer(r"\S+", name="whitespace")
        tokens = tokenizer("hello, world!")
        assert tokens[0].raw == "hello,"
        assert tokens[1].raw == "world!"
