"""Tests for bewer.core.keyword module."""

import warnings

from bewer.core.keyword import Keyword, KeywordNotFoundWarning, KeywordTrie
from bewer.core.text import Text, TextType, TokenList


class TestKeywordInit:
    """Tests for Keyword initialization."""

    def test_is_subclass_of_text(self):
        """Test that Keyword is a subclass of Text."""
        assert issubclass(Keyword, Text)

    def test_text_type_is_keyword(self, sample_dataset):
        """Test that Keyword always has TextType.KEYWORD."""
        sample_dataset.add("hello world", "hello world", keywords={"greetings": ["hello"]})
        kw = next(iter(sample_dataset[-1].keywords["greetings"]))
        assert kw.text_type == TextType.KEYWORD

    def test_isinstance_text(self, sample_dataset):
        """Test that Keyword instances are also Text instances."""
        sample_dataset.add("hello world", "hello world", keywords={"greetings": ["hello"]})
        kw = next(iter(sample_dataset[-1].keywords["greetings"]))
        assert isinstance(kw, Text)
        assert isinstance(kw, Keyword)


class TestKeywordProperties:
    """Tests for inherited standardized and tokens properties."""

    def test_standardized(self, sample_dataset):
        """Test that standardized property works on Keyword."""
        sample_dataset.add("hello world", "hello world", keywords={"greetings": ["hello"]})
        kw = next(iter(sample_dataset[-1].keywords["greetings"]))
        assert isinstance(kw.standardized, str)

    def test_tokens(self, sample_dataset):
        """Test that tokens property works on Keyword."""
        sample_dataset.add("hello world", "hello world", keywords={"greetings": ["hello"]})
        kw = next(iter(sample_dataset[-1].keywords["greetings"]))
        assert isinstance(kw.tokens, TokenList)
        assert len(kw.tokens) == 1


class TestKeywordTrieFindInTokens:
    """Tests for KeywordTrie.find_in_tokens() method."""

    def test_single_token_found(self, sample_dataset):
        """Test finding a single-token keyword in a token list."""
        sample_dataset.add("the quick brown fox", "the quick brown dog", keywords={"animals": ["fox"]})
        example = sample_dataset[-1]
        trie = KeywordTrie(example.keywords["animals"])
        ref_tokens = example.ref.tokens
        matches = trie.find_in_tokens(ref_tokens)
        assert len(matches) == 1
        assert ref_tokens[matches[0]][0].raw == "fox"

    def test_single_token_multiple_occurrences(self, sample_dataset):
        """Test finding a token that appears multiple times."""
        sample_dataset.add("the fox and the fox", "the fox", keywords={"animals": ["fox"]})
        example = sample_dataset[-1]
        trie = KeywordTrie(example.keywords["animals"])
        matches = trie.find_in_tokens(example.ref.tokens)
        assert len(matches) == 2

    def test_multi_token_keyword(self, sample_dataset):
        """Test finding a multi-token keyword contiguously."""
        sample_dataset.add("the quick brown fox", "the quick dog", keywords={"phrases": ["quick brown"]})
        example = sample_dataset[-1]
        trie = KeywordTrie(example.keywords["phrases"])
        ref_tokens = example.ref.tokens
        matches = trie.find_in_tokens(ref_tokens)
        assert len(matches) == 1
        matched_tokens = ref_tokens[matches[0]]
        assert len(matched_tokens) == 2
        assert matched_tokens.raw == ["quick", "brown"]

    def test_no_match(self, sample_dataset):
        """Test that non-matching keyword returns empty list."""
        sample_dataset.add("hello world", "hello world", keywords={"greetings": ["hello"]})
        example = sample_dataset[-1]
        trie = KeywordTrie(example.keywords["greetings"])
        # Search in a different example's tokens where "hello" doesn't appear
        other_tokens = sample_dataset[1].ref.tokens  # "the quick brown fox"
        matches = trie.find_in_tokens(other_tokens)
        assert len(matches) == 0

    def test_returns_slices(self, sample_dataset):
        """Test that matches are slice instances."""
        sample_dataset.add("the quick brown fox", "the quick", keywords={"colors": ["brown"]})
        example = sample_dataset[-1]
        trie = KeywordTrie(example.keywords["colors"])
        matches = trie.find_in_tokens(example.ref.tokens)
        assert all(isinstance(m, slice) for m in matches)

    def test_warn_missing(self, sample_dataset):
        """Test that warn_missing emits KeywordNotFoundWarning for unmatched keywords."""
        sample_dataset.add("hello world", "hello world", keywords={"missing": ["nonexistent"]})
        example = sample_dataset[-1]
        trie = KeywordTrie(example.keywords["missing"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            matches = trie.find_in_tokens(example.ref.tokens, warn_missing=True)
        assert len(matches) == 0
        assert len([x for x in w if issubclass(x.category, KeywordNotFoundWarning)]) > 0


class TestKeywordRepr:
    """Tests for Keyword.__repr__()."""

    def test_repr(self, sample_dataset):
        """Test that repr shows Keyword prefix."""
        sample_dataset.add("hello world", "hello world", keywords={"greetings": ["hello"]})
        kw = next(iter(sample_dataset[-1].keywords["greetings"]))
        assert "Keyword" in repr(kw)
        assert "hello" in repr(kw)


class TestTokenListIndices:
    """Tests for TokenList.indices() method."""

    def test_single_match(self, sample_dataset):
        """Test finding a single matching token."""
        tokens = sample_dataset[1].ref.tokens  # "the quick brown fox"
        indices = tokens.indices("fox")
        assert indices == {3}

    def test_no_match(self, sample_dataset):
        """Test that non-matching text returns empty set."""
        tokens = sample_dataset[1].ref.tokens  # "the quick brown fox"
        indices = tokens.indices("nonexistent")
        assert indices == set()

    def test_multiple_matches(self, sample_dataset):
        """Test finding multiple matching tokens."""
        sample_dataset.add("the fox and the fox", "the fox")
        tokens = sample_dataset[-1].ref.tokens
        indices = tokens.indices("the")
        assert indices == {0, 3}

    def test_raw_mode(self, sample_dataset):
        """Test using raw text for comparison."""
        tokens = sample_dataset[1].ref.tokens  # "the quick brown fox"
        indices = tokens.indices("fox", normalized=False)
        assert indices == {3}
