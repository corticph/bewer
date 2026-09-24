"""Tests for bewer.core.token module."""

import regex as re

from bewer import Dataset
from bewer.core.token import Token


class TestTokenInit:
    """Tests for Token.__init__()."""

    def test_basic_initialization(self, pipelines, stub_parent):
        """Test basic token initialization."""
        token = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        assert token.standardized == "hello"
        assert token.start == 0
        assert token.end == 5

    def test_with_index(self, pipelines, stub_parent):
        """Test token initialization with index."""
        token = Token(standardized="world", start=6, end=11, index=1, pipelines=pipelines, src=stub_parent)
        assert token.index == 1

    def test_slice_property(self, pipelines, stub_parent):
        """Test that slice property is correctly set."""
        token = Token(standardized="test", start=10, end=14, pipelines=pipelines, src=stub_parent)
        assert token.slice == slice(10, 14)

    def test_default_index_none(self, pipelines, stub_parent):
        """Test that index defaults to None."""
        token = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        assert token.index is None

    def test_normalized_cache_initialized(self, pipelines, stub_parent):
        """Test that normalized cache is initialized as empty dict."""
        token = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        assert token._cache_normalized == {}

    def test_src_is_stored(self, pipelines, stub_parent):
        """Test that the provided src is stored."""
        token = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        assert token.src is stub_parent

    def test_src_defaults_to_none(self, pipelines):
        """Test that src is optional and defaults to None."""
        token = Token(standardized="hello", start=0, end=5, pipelines=pipelines)
        assert token.src is None


class TestTokenFromMatch:
    """Tests for Token.from_match()."""

    def test_basic_from_match(self, pipelines, stub_parent):
        """Test creating token from regex match."""
        pattern = re.compile(r"\S+")
        match = pattern.search("hello world", 0)
        token = Token.from_match(match, index=0, pipelines=pipelines, src=stub_parent)
        assert token.standardized == "hello"
        assert token.start == 0
        assert token.end == 5
        assert token.index == 0

    def test_from_match_middle_of_string(self, pipelines, stub_parent):
        """Test creating token from match in middle of string."""
        pattern = re.compile(r"\S+")
        text = "hello world"
        matches = list(pattern.finditer(text))
        token = Token.from_match(matches[1], index=1, pipelines=pipelines, src=stub_parent)
        assert token.standardized == "world"
        assert token.start == 6
        assert token.end == 11

    def test_from_match_stores_src(self, pipelines, stub_parent):
        """Test creating token with source text reference."""
        pattern = re.compile(r"\S+")
        match = pattern.search("hello")
        token = Token.from_match(match, index=0, pipelines=pipelines, src=stub_parent)
        assert token.src is stub_parent


class TestTokenEquality:
    """Tests for Token.__eq__()."""

    def test_equal_tokens(self, pipelines, stub_parent):
        """Test that identical tokens are equal."""
        token1 = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        token2 = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        assert token1 == token2

    def test_different_standardized(self, pipelines, stub_parent):
        """Test tokens with different standardized values are not equal."""
        token1 = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        token2 = Token(standardized="world", start=0, end=5, pipelines=pipelines, src=stub_parent)
        assert token1 != token2

    def test_different_start(self, pipelines, stub_parent):
        """Test tokens with different start positions are not equal."""
        token1 = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        token2 = Token(standardized="hello", start=1, end=5, pipelines=pipelines, src=stub_parent)
        assert token1 != token2

    def test_different_end(self, pipelines, stub_parent):
        """Test tokens with different end positions are not equal."""
        token1 = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        token2 = Token(standardized="hello", start=0, end=6, pipelines=pipelines, src=stub_parent)
        assert token1 != token2

    def test_comparison_with_non_token(self, pipelines, stub_parent):
        """Test comparison with non-Token objects returns False."""
        token = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        assert token != "hello"
        assert token != 42
        assert token is not None

    def test_index_not_considered_in_equality(self, pipelines, stub_parent):
        """Test that index is not considered in equality."""
        token1 = Token(standardized="hello", start=0, end=5, index=0, pipelines=pipelines, src=stub_parent)
        token2 = Token(standardized="hello", start=0, end=5, index=1, pipelines=pipelines, src=stub_parent)
        assert token1 == token2


class TestTokenInctx:
    """Tests for Token.inctx() context extraction."""

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

    def test_inctx_without_src_returns_standardized(self, pipelines):
        """A standalone token (no parent) falls back to its standardized text."""
        token = Token(standardized="hello", start=0, end=5, pipelines=pipelines)
        assert token.inctx() == "hello"

    def test_inctx_slices_the_standardized_string(self):
        """The context window is the exact standardized slice the offsets designate.

        NFC composes "e" + U+0301 into a single "é", so the standardized string is
        shorter than the raw one and slicing ``Text.raw`` with these offsets drifts by
        the length delta. Asserting the exact slice (rather than mere containment of
        the token) is what pins the coordinate system: a drifted window still contains
        its token as a substring, so containment alone would not catch the regression.
        """
        text = self._nfc_text()
        standardized = text.standardized
        assert len(standardized) < len(text.raw)

        width = 8
        for token in text.tokens:
            start = max(0, token.start - width)
            end = min(len(standardized), token.end + width)
            assert token.inctx(width=width, add_ellipsis=False) == standardized[start:end]

    def test_inctx_highlight_wraps_exactly_the_token(self):
        """The highlighted region covers the token itself, not a drifted span."""
        text = self._nfc_text()

        for token in text.tokens:
            ctx = token.inctx(width=8, highlight=True, add_ellipsis=False)
            styled = re.findall(r"\x1b\[[0-9;]*m(.+?)\x1b\[0m", ctx)
            assert styled == [token.standardized], f"{styled!r} != [{token.standardized!r}]"

    @staticmethod
    def _nfc_text():
        """A Text whose standardization (NFC) shortens the string by three characters."""
        dataset = Dataset(language="en")
        source = "cafe\u0301 latte and re\u0301sume\u0301 words here"
        dataset.add(ref=source, hyp=source)
        return dataset[0].ref


class TestTokenRepr:
    """Tests for Token.__repr__()."""

    def test_repr(self, pipelines, stub_parent):
        """Test string representation."""
        token = Token(standardized="hello", start=0, end=5, pipelines=pipelines, src=stub_parent)
        assert repr(token) == 'Token("hello")'

    def test_repr_with_special_chars(self, pipelines, stub_parent):
        """Test repr with special characters in token."""
        token = Token(standardized="hello!", start=0, end=6, pipelines=pipelines, src=stub_parent)
        assert repr(token) == 'Token("hello!")'
