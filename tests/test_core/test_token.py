"""Tests for bewer.core.token module."""

import pytest
import regex as re

from bewer.core.token import Token


class TestTokenInit:
    """Tests for Token.__init__()."""

    def test_basic_initialization(self):
        """Test basic token initialization."""
        token = Token(raw="hello", start=0, end=5)
        assert token.raw == "hello"
        assert token.start == 0
        assert token.end == 5

    def test_with_index(self):
        """Test token initialization with index."""
        token = Token(raw="world", start=6, end=11, index=1)
        assert token.index == 1

    def test_slice_property(self):
        """Test that slice property is correctly set."""
        token = Token(raw="test", start=10, end=14)
        assert token.slice == slice(10, 14)

    def test_default_index_none(self):
        """Test that index defaults to None."""
        token = Token(raw="hello", start=0, end=5)
        assert token.index is None

    def test_normalized_cache_initialized(self):
        """Test that normalized cache is initialized as empty dict."""
        token = Token(raw="hello", start=0, end=5)
        assert token._cache_normalized == {}

    def test_src_none_by_default(self):
        """Test that src defaults to None."""
        token = Token(raw="hello", start=0, end=5)
        assert token.src is None


class TestTokenFromMatch:
    """Tests for Token.from_match()."""

    def test_basic_from_match(self):
        """Test creating token from regex match."""
        pattern = re.compile(r"\S+")
        match = pattern.search("hello world", 0)
        token = Token.from_match(match, index=0)
        assert token.raw == "hello"
        assert token.start == 0
        assert token.end == 5
        assert token.index == 0

    def test_from_match_middle_of_string(self):
        """Test creating token from match in middle of string."""
        pattern = re.compile(r"\S+")
        text = "hello world"
        matches = list(pattern.finditer(text))
        token = Token.from_match(matches[1], index=1)
        assert token.raw == "world"
        assert token.start == 6
        assert token.end == 11

    def test_from_match_with_src(self):
        """Test creating token with source text reference."""
        pattern = re.compile(r"\S+")
        match = pattern.search("hello")
        token = Token.from_match(match, index=0, src=None)
        assert token.src is None


class TestTokenEquality:
    """Tests for Token.__eq__()."""

    def test_equal_tokens(self):
        """Test that identical tokens are equal."""
        token1 = Token(raw="hello", start=0, end=5)
        token2 = Token(raw="hello", start=0, end=5)
        assert token1 == token2

    def test_different_raw(self):
        """Test tokens with different raw values are not equal."""
        token1 = Token(raw="hello", start=0, end=5)
        token2 = Token(raw="world", start=0, end=5)
        assert token1 != token2

    def test_different_start(self):
        """Test tokens with different start positions are not equal."""
        token1 = Token(raw="hello", start=0, end=5)
        token2 = Token(raw="hello", start=1, end=5)
        assert token1 != token2

    def test_different_end(self):
        """Test tokens with different end positions are not equal."""
        token1 = Token(raw="hello", start=0, end=5)
        token2 = Token(raw="hello", start=0, end=6)
        assert token1 != token2

    def test_comparison_with_non_token(self):
        """Test comparison with non-Token objects returns False."""
        token = Token(raw="hello", start=0, end=5)
        assert token != "hello"
        assert token != 42
        assert token is not None

    def test_index_not_considered_in_equality(self):
        """Test that index is not considered in equality."""
        token1 = Token(raw="hello", start=0, end=5, index=0)
        token2 = Token(raw="hello", start=0, end=5, index=1)
        assert token1 == token2


class TestTokenInctx:
    """Tests for Token.inctx() context extraction."""

    def test_inctx_raises_without_src_text(self):
        """Test that inctx raises ValueError without source text."""
        token = Token(raw="hello", start=0, end=5)
        with pytest.raises(ValueError, match="Source text is not set"):
            token.inctx()

    def test_inctx_with_dataset_context(self, sample_dataset):
        """Test inctx with proper context from dataset."""
        example = sample_dataset[1]  # "the quick brown fox"
        tokens = example.ref.tokens
        token = tokens[1]  # "quick"

        ctx = token.inctx(width=5, highlight=False, add_ellipsis=True)
        assert "quick" in ctx

    def test_inctx_without_ellipsis(self, sample_dataset):
        """Test inctx without ellipsis."""
        example = sample_dataset[0]  # "hello world"
        tokens = example.ref.tokens
        token = tokens[0]  # "hello"

        # With large enough width, no ellipsis needed
        ctx = token.inctx(width=100, highlight=False, add_ellipsis=True)
        assert not ctx.startswith("...")
        assert "hello" in ctx


class TestTokenRepr:
    """Tests for Token.__repr__()."""

    def test_repr(self):
        """Test string representation."""
        token = Token(raw="hello", start=0, end=5)
        assert repr(token) == 'Token("hello")'

    def test_repr_with_special_chars(self):
        """Test repr with special characters in token."""
        token = Token(raw="hello!", start=0, end=6)
        assert repr(token) == 'Token("hello!")'
