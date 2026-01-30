"""Tests for bewer.core.text module."""

import pytest

from bewer.core.text import Text, TextType, TokenList


class TestTextRaw:
    """Tests for Text.raw property."""

    def test_raw_returns_original_text(self, sample_text):
        """Test that raw property returns the original text."""
        assert sample_text.raw == "hello world"

    def test_raw_raises_when_none(self):
        """Test that raw raises ValueError when _raw is None."""
        text = Text(raw=None)
        with pytest.raises(ValueError, match="Raw text is None"):
            _ = text.raw


class TestTextStandardized:
    """Tests for Text.standardized property."""

    def test_standardized_with_pipeline(self, sample_text):
        """Test standardized text with pipeline."""
        # The standardized property uses the pipeline from the dataset
        standardized = sample_text.standardized
        assert isinstance(standardized, str)

    def test_standardized_raises_without_pipeline(self):
        """Test that standardized raises ValueError without pipeline."""
        text = Text(raw="hello")
        with pytest.raises(ValueError, match="No standardizers found"):
            _ = text.standardized


class TestTextTokens:
    """Tests for Text.tokens property."""

    def test_tokens_returns_tokenlist(self, sample_text):
        """Test that tokens property returns a TokenList."""
        tokens = sample_text.tokens
        assert isinstance(tokens, TokenList)

    def test_tokens_correct_count(self, sample_text):
        """Test that tokenization produces correct number of tokens."""
        tokens = sample_text.tokens
        assert len(tokens) == 2  # "hello" and "world"

    def test_tokens_raises_without_pipeline(self):
        """Test that tokens raises ValueError without pipeline."""
        text = Text(raw="hello world")
        with pytest.raises(ValueError):
            _ = text.tokens


class TestTextJoined:
    """Tests for Text.joined() method."""

    def test_joined_normalized(self, sample_text):
        """Test joining tokens with normalization."""
        joined = sample_text.joined(normalized=True)
        assert isinstance(joined, str)
        # The joined text should contain the tokens
        assert "hello" in joined.lower()
        assert "world" in joined.lower()

    def test_joined_raw(self, sample_text):
        """Test joining tokens without normalization."""
        joined = sample_text.joined(normalized=False)
        assert isinstance(joined, str)


class TestTextType:
    """Tests for TextType enum."""

    def test_text_type_values(self):
        """Test TextType enum values."""
        assert TextType.REF == "ref"
        assert TextType.HYP == "hyp"
        assert TextType.KEYWORD == "keyword"


class TestTextRepr:
    """Tests for Text.__repr__()."""

    def test_repr_short_text(self, sample_dataset):
        """Test repr with short text."""
        text = sample_dataset[0].ref  # "hello world"
        repr_str = repr(text)
        assert "hello world" in repr_str
        assert "Text" in repr_str

    def test_repr_long_text(self, sample_dataset):
        """Test repr truncates long text."""
        # Create a dataset with long text
        dataset = sample_dataset
        dataset.add("a" * 100, "b" * 100)
        text = dataset[-1].ref
        repr_str = repr(text)
        assert "..." in repr_str
        assert len(repr_str) < 60


class TestTextHash:
    """Tests for Text.__hash__()."""

    def test_hash_same_text_same_type(self, sample_dataset):
        """Test that same text with same type has same hash."""
        # Two texts with same content should have same hash if same type
        text1 = sample_dataset[0].ref
        text2 = sample_dataset[0].ref
        assert hash(text1) == hash(text2)


class TestTokenListRaw:
    """Tests for TokenList.raw property."""

    def test_raw_returns_list_of_strings(self, sample_tokens):
        """Test that raw property returns list of strings."""
        raw = sample_tokens.raw
        assert isinstance(raw, list)
        assert all(isinstance(t, str) for t in raw)
        assert raw == ["hello", "world"]


class TestTokenListNormalized:
    """Tests for TokenList.normalized property."""

    def test_normalized_returns_list_of_strings(self, sample_tokens):
        """Test that normalized property returns list of strings."""
        normalized = sample_tokens.normalized
        assert isinstance(normalized, list)
        assert all(isinstance(t, str) for t in normalized)


class TestTokenListNgrams:
    """Tests for TokenList.ngrams() method."""

    def test_unigrams(self, sample_dataset):
        """Test extracting unigrams (n=1)."""
        tokens = sample_dataset[1].ref.tokens  # "the quick brown fox"
        ngrams = tokens.ngrams(n=1, normalized=True)
        assert len(ngrams) == 4

    def test_bigrams(self, sample_dataset):
        """Test extracting bigrams (n=2)."""
        tokens = sample_dataset[1].ref.tokens  # "the quick brown fox"
        ngrams = tokens.ngrams(n=2, normalized=True, join_tokens=True)
        assert len(ngrams) == 3  # 4 tokens -> 3 bigrams

    def test_trigrams(self, sample_dataset):
        """Test extracting trigrams (n=3)."""
        tokens = sample_dataset[1].ref.tokens  # "the quick brown fox"
        ngrams = tokens.ngrams(n=3, normalized=True, join_tokens=True)
        assert len(ngrams) == 2  # 4 tokens -> 2 trigrams

    def test_ngrams_invalid_n(self, sample_tokens):
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n must be a positive integer"):
            sample_tokens.ngrams(n=0)

    def test_ngrams_n_larger_than_list(self, sample_tokens):
        """Test n-grams when n is larger than token list."""
        ngrams = sample_tokens.ngrams(n=10, normalized=True)
        assert len(ngrams) == 0


class TestTokenListSlicing:
    """Tests for TokenList slicing and iteration."""

    def test_getitem_returns_token(self, sample_tokens):
        """Test that indexing returns a Token."""
        token = sample_tokens[0]
        assert token.raw == "hello"

    def test_slice_returns_tokenlist(self, sample_tokens):
        """Test that slicing returns a TokenList."""
        sliced = sample_tokens[0:1]
        assert isinstance(sliced, TokenList)
        assert len(sliced) == 1

    def test_add_returns_tokenlist(self, sample_dataset):
        """Test that adding TokenLists returns TokenList."""
        tokens1 = sample_dataset[0].ref.tokens
        tokens2 = sample_dataset[1].ref.tokens
        combined = tokens1 + tokens2
        assert isinstance(combined, TokenList)
        assert len(combined) == len(tokens1) + len(tokens2)


class TestTokenListRepr:
    """Tests for TokenList.__repr__()."""

    def test_repr_short_list(self, sample_tokens):
        """Test repr with short token list."""
        repr_str = repr(sample_tokens)
        assert "TokenList" in repr_str
        assert "hello" in repr_str
        assert "world" in repr_str
