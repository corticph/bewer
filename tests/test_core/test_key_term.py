"""Tests for bewer.core.key_term module."""

import warnings

from bewer.core.key_term import KeyTerm, KeyTermNotFoundWarning, KeyTermTrie, _remove_subset_matches
from bewer.core.text import Text, TextType, TokenList


class TestKeyTermInit:
    """Tests for KeyTerm initialization."""

    def test_is_subclass_of_text(self):
        """Test that KeyTerm is a subclass of Text."""
        assert issubclass(KeyTerm, Text)

    def test_text_type_is_key_term(self, sample_dataset):
        """Test that KeyTerm always has TextType.KEY_TERM."""
        sample_dataset.add("hello world", "hello world", key_terms={"greetings": ["hello"]})
        kt = next(iter(sample_dataset[-1].key_terms["greetings"]))
        assert kt.text_type == TextType.KEY_TERM

    def test_isinstance_text(self, sample_dataset):
        """Test that KeyTerm instances are also Text instances."""
        sample_dataset.add("hello world", "hello world", key_terms={"greetings": ["hello"]})
        kt = next(iter(sample_dataset[-1].key_terms["greetings"]))
        assert isinstance(kt, Text)
        assert isinstance(kt, KeyTerm)


class TestKeyTermProperties:
    """Tests for inherited standardized and tokens properties."""

    def test_standardized(self, sample_dataset):
        """Test that standardized property works on KeyTerm."""
        sample_dataset.add("hello world", "hello world", key_terms={"greetings": ["hello"]})
        kt = next(iter(sample_dataset[-1].key_terms["greetings"]))
        assert isinstance(kt.standardized, str)

    def test_tokens(self, sample_dataset):
        """Test that tokens property works on KeyTerm."""
        sample_dataset.add("hello world", "hello world", key_terms={"greetings": ["hello"]})
        kt = next(iter(sample_dataset[-1].key_terms["greetings"]))
        assert isinstance(kt.tokens, TokenList)
        assert len(kt.tokens) == 1


class TestKeyTermTrieFindInTokens:
    """Tests for KeyTermTrie.find_in_tokens() method."""

    def test_single_token_found(self, sample_dataset):
        """Test finding a single-token key term in a token list."""
        sample_dataset.add("the quick brown fox", "the quick brown dog", key_terms={"animals": ["fox"]})
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["animals"])
        ref_tokens = example.ref.tokens
        matches = trie.find_in_tokens(ref_tokens)
        assert len(matches) == 1
        assert ref_tokens[matches[0]][0].raw == "fox"

    def test_single_token_multiple_occurrences(self, sample_dataset):
        """Test finding a token that appears multiple times."""
        sample_dataset.add("the fox and the fox", "the fox", key_terms={"animals": ["fox"]})
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["animals"])
        matches = trie.find_in_tokens(example.ref.tokens)
        assert len(matches) == 2

    def test_multi_token_key_term(self, sample_dataset):
        """Test finding a multi-token key term contiguously."""
        sample_dataset.add("the quick brown fox", "the quick dog", key_terms={"phrases": ["quick brown"]})
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["phrases"])
        ref_tokens = example.ref.tokens
        matches = trie.find_in_tokens(ref_tokens)
        assert len(matches) == 1
        matched_tokens = ref_tokens[matches[0]]
        assert len(matched_tokens) == 2
        assert matched_tokens.raw == ["quick", "brown"]

    def test_no_match(self, sample_dataset):
        """Test that non-matching key term returns empty list."""
        sample_dataset.add("hello world", "hello world", key_terms={"greetings": ["hello"]})
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["greetings"])
        # Search in a different example's tokens where "hello" doesn't appear
        other_tokens = sample_dataset[1].ref.tokens  # "the quick brown fox"
        matches = trie.find_in_tokens(other_tokens)
        assert len(matches) == 0

    def test_returns_slices(self, sample_dataset):
        """Test that matches are slice instances."""
        sample_dataset.add("the quick brown fox", "the quick", key_terms={"colors": ["brown"]})
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["colors"])
        matches = trie.find_in_tokens(example.ref.tokens)
        assert all(isinstance(m, slice) for m in matches)

    def test_warn_missing(self, sample_dataset):
        """Test that warn_missing emits KeyTermNotFoundWarning for unmatched key terms."""
        sample_dataset.add("hello world", "hello world", key_terms={"missing": ["nonexistent"]})
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["missing"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            matches = trie.find_in_tokens(example.ref.tokens, warn_missing=True)
        assert len(matches) == 0
        assert len([x for x in w if issubclass(x.category, KeyTermNotFoundWarning)]) > 0


class TestKeyTermRepr:
    """Tests for KeyTerm.__repr__()."""

    def test_repr(self, sample_dataset):
        """Test that repr shows KeyTerm prefix."""
        sample_dataset.add("hello world", "hello world", key_terms={"greetings": ["hello"]})
        kt = next(iter(sample_dataset[-1].key_terms["greetings"]))
        assert "KeyTerm" in repr(kt)
        assert "hello" in repr(kt)


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


class TestRemoveSubsetMatches:
    """Tests for _remove_subset_matches() function."""

    def test_empty_input(self):
        assert _remove_subset_matches([]) == []

    def test_no_overlaps(self):
        matches = [slice(0, 1), slice(2, 3), slice(5, 7)]
        result = _remove_subset_matches(matches)
        assert len(result) == 3

    def test_subset_removed(self):
        """A shorter match contained within a longer match is removed."""
        matches = [slice(1, 4), slice(1, 3)]
        result = _remove_subset_matches(matches)
        assert result == [slice(1, 4)]

    def test_adjacent_kept(self):
        """Adjacent non-overlapping matches are preserved."""
        matches = [slice(0, 2), slice(2, 4)]
        result = _remove_subset_matches(matches)
        assert len(result) == 2

    def test_identical_deduplicated(self):
        """Identical matches are deduplicated to one."""
        matches = [slice(1, 3), slice(1, 3)]
        result = _remove_subset_matches(matches)
        assert result == [slice(1, 3)]


class TestKeyTermTrieAllowSubsets:
    """Tests for KeyTermTrie.find_in_tokens() with allow_subsets parameter."""

    def test_allow_subsets_true_returns_all(self, sample_dataset):
        """With allow_subsets=True (default), overlapping key terms both match."""
        sample_dataset.add(
            "the quick brown fox",
            "the quick brown fox",
            key_terms={"phrases": ["quick", "quick brown"]},
        )
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["phrases"])
        matches = trie.find_in_tokens(example.ref.tokens, allow_subsets=True)
        assert len(matches) == 2

    def test_allow_subsets_false_keeps_longer(self, sample_dataset):
        """With allow_subsets=False, the shorter overlapping match is removed."""
        sample_dataset.add(
            "the quick brown fox",
            "the quick brown fox",
            key_terms={"phrases": ["quick", "quick brown"]},
        )
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["phrases"])
        matches = trie.find_in_tokens(example.ref.tokens, allow_subsets=False)
        assert len(matches) == 1
        matched = example.ref.tokens[matches[0]]
        assert matched.raw == ["quick", "brown"]


class TestKeyTermTrieAddCapitalized:
    """Tests for KeyTermTrie with add_capitalized parameter."""

    def test_add_capitalized_matches_sentence_start(self, sample_dataset):
        """With normalized=False and add_capitalized=True, matches capitalized variant."""
        sample_dataset.add("Hello world", "Hello world", key_terms={"greetings": ["hello"]})
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["greetings"], normalized=False, add_capitalized=True)
        matches = trie.find_in_tokens(example.ref.tokens)
        assert len(matches) == 1

    def test_no_capitalized_misses_sentence_start(self, sample_dataset):
        """With normalized=False and add_capitalized=False, does not match capitalized text."""
        sample_dataset.add("Hello world", "Hello world", key_terms={"greetings": ["hello"]})
        example = sample_dataset[-1]
        trie = KeyTermTrie(example.key_terms["greetings"], normalized=False, add_capitalized=False)
        matches = trie.find_in_tokens(example.ref.tokens)
        assert len(matches) == 0


class TestTokenListSrc:
    """Tests for TokenList.src property."""

    def test_src_from_text_tokens(self, sample_dataset):
        """Text.tokens produces a TokenList whose src points to the Text."""
        text = sample_dataset[0].ref
        tokens = text.tokens
        assert tokens.src is text

    def test_default_src_is_none(self):
        """A bare TokenList() has src == None."""
        tokens = TokenList()
        assert tokens.src is None


class TestKeyTermNotFoundWarningImport:
    """Tests for KeyTermNotFoundWarning import paths."""

    def test_importable_from_key_term_module(self):
        from bewer.core.key_term import KeyTermNotFoundWarning as W1

        assert issubclass(W1, UserWarning)

    def test_importable_from_top_level(self):
        from bewer import KeyTermNotFoundWarning as W2

        assert issubclass(W2, UserWarning)

    def test_same_class(self):
        from bewer import KeyTermNotFoundWarning as W1
        from bewer.core.key_term import KeyTermNotFoundWarning as W2

        assert W1 is W2
