"""Tests for bewer.core.key_term module."""

from bewer import Vocabulary
from bewer.core.key_term import KeyTerm, KeyTermMatch, KeyTermTrie, _remove_subset_matches
from bewer.core.text import Text, TextType, TokenList


class TestKeyTermInit:
    """Tests for KeyTerm initialization."""

    def test_is_subclass_of_text(self):
        """Test that KeyTerm is a subclass of Text."""
        assert issubclass(KeyTerm, Text)

    def test_text_type_is_key_term(self, sample_dataset):
        """Test that KeyTerm always has TextType.KEY_TERM."""
        kt = KeyTerm("hello", pipelines=sample_dataset.pipelines)
        assert kt.text_type == TextType.KEY_TERM

    def test_isinstance_text(self, sample_dataset):
        """Test that KeyTerm instances are also Text instances."""
        kt = KeyTerm("hello", pipelines=sample_dataset.pipelines)
        assert isinstance(kt, Text)
        assert isinstance(kt, KeyTerm)


class TestKeyTermProperties:
    """Tests for inherited standardized and tokens properties."""

    def test_standardized(self, sample_dataset):
        """Test that standardized property works on KeyTerm."""
        kt = KeyTerm("hello", pipelines=sample_dataset.pipelines)
        assert isinstance(kt.standardized, str)

    def test_tokens(self, sample_dataset):
        """Test that tokens property works on KeyTerm."""
        kt = KeyTerm("hello", pipelines=sample_dataset.pipelines)
        assert isinstance(kt.tokens, TokenList)
        assert len(kt.tokens) == 1


class TestKeyTermTrieFindInTokens:
    """Tests for KeyTermTrie.find_in_tokens() method."""

    def test_single_token_found(self, sample_dataset):
        """Test finding a single-token key term in a token list."""
        sample_dataset.add("the quick brown fox", "the quick brown dog")
        example = sample_dataset[-1]
        trie = KeyTermTrie({KeyTerm("fox", pipelines=sample_dataset.pipelines)})
        ref_tokens = example.ref.tokens
        matches, _ = trie.find_in_tokens(ref_tokens)
        assert len(matches) == 1
        assert ref_tokens[matches[0]][0].raw == "fox"

    def test_single_token_multiple_occurrences(self, sample_dataset):
        """Test finding a token that appears multiple times."""
        sample_dataset.add("the fox and the fox", "the fox")
        example = sample_dataset[-1]
        trie = KeyTermTrie({KeyTerm("fox", pipelines=sample_dataset.pipelines)})
        matches, _ = trie.find_in_tokens(example.ref.tokens)
        assert len(matches) == 2

    def test_multi_token_key_term(self, sample_dataset):
        """Test finding a multi-token key term contiguously."""
        sample_dataset.add("the quick brown fox", "the quick dog")
        example = sample_dataset[-1]
        trie = KeyTermTrie({KeyTerm("quick brown", pipelines=sample_dataset.pipelines)})
        ref_tokens = example.ref.tokens
        matches, _ = trie.find_in_tokens(ref_tokens)
        assert len(matches) == 1
        matched_tokens = ref_tokens[matches[0]]
        assert len(matched_tokens) == 2
        assert matched_tokens.raw == ["quick", "brown"]

    def test_no_match(self, sample_dataset):
        """Test that non-matching key term returns empty list."""
        sample_dataset.add("hello world", "hello world")
        trie = KeyTermTrie({KeyTerm("hello", pipelines=sample_dataset.pipelines)})
        # Search in a different example's tokens where "hello" doesn't appear
        other_tokens = sample_dataset[1].ref.tokens  # "the quick brown fox"
        matches, _ = trie.find_in_tokens(other_tokens)
        assert len(matches) == 0

    def test_returns_slices(self, sample_dataset):
        """Test that matches are slice instances."""
        sample_dataset.add("the quick brown fox", "the quick")
        example = sample_dataset[-1]
        trie = KeyTermTrie({KeyTerm("brown", pipelines=sample_dataset.pipelines)})
        matches, _ = trie.find_in_tokens(example.ref.tokens)
        assert all(isinstance(m, slice) for m in matches)

    def test_matched_key_term_returned(self, sample_dataset):
        """find_in_tokens returns the KeyTerm object that produced each match."""
        sample_dataset.add("hello world", "hello world")
        example = sample_dataset[-1]
        kt = KeyTerm("hello", pipelines=sample_dataset.pipelines)
        trie = KeyTermTrie({kt})
        _, key_terms = trie.find_in_tokens(example.ref.tokens)
        assert key_terms == [kt]

    def test_no_key_terms_when_unmatched(self, sample_dataset):
        """find_in_tokens returns no key terms when nothing matches."""
        sample_dataset.add("hello world", "hello world")
        example = sample_dataset[-1]
        kt = KeyTerm("nonexistent", pipelines=sample_dataset.pipelines)
        trie = KeyTermTrie({kt})
        _, key_terms = trie.find_in_tokens(example.ref.tokens)
        assert key_terms == []


class TestKeyTermRepr:
    """Tests for KeyTerm.__repr__()."""

    def test_repr(self, sample_dataset):
        """Test that repr shows KeyTerm prefix."""
        kt = KeyTerm("hello", pipelines=sample_dataset.pipelines)
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


class TestTextGetKeyTermMatchesAllowSubsets:
    """Tests for Text.get_key_term_matches() with allow_subset_matches parameter."""

    def test_allow_subset_matches_true_returns_all(self, sample_dataset):
        """With allow_subset_matches=True (default), overlapping key terms both match."""
        sample_dataset.add(
            "the quick brown fox",
            "the quick brown fox",
        )
        sample_dataset.add_vocabulary(Vocabulary(name="phrases").add_terms(["quick", "quick brown"]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="phrases", allow_subset_matches=True)
        assert len(matches) == 2

    def test_allow_subset_matches_false_keeps_longer(self, sample_dataset):
        """With allow_subset_matches=False, the shorter overlapping match is removed."""
        sample_dataset.add(
            "the quick brown fox",
            "the quick brown fox",
        )
        sample_dataset.add_vocabulary(Vocabulary(name="phrases").add_terms(["quick", "quick brown"]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="phrases", allow_subset_matches=False)
        assert len(matches) == 1
        assert matches[0].tokens.raw == ["quick", "brown"]


class TestKeyTermTrieAddCapitalized:
    """Tests for KeyTermTrie with add_capitalized parameter."""

    def test_add_capitalized_matches_sentence_start(self, sample_dataset):
        """With normalized=False and add_capitalized=True, matches capitalized variant."""
        sample_dataset.add("Hello world", "Hello world")
        example = sample_dataset[-1]
        kt = KeyTerm("hello", pipelines=sample_dataset.pipelines)
        trie = KeyTermTrie({kt}, normalized=False, add_capitalized=True)
        matches, _ = trie.find_in_tokens(example.ref.tokens)
        assert len(matches) == 1

    def test_no_capitalized_misses_sentence_start(self, sample_dataset):
        """With normalized=False and add_capitalized=False, does not match capitalized text."""
        sample_dataset.add("Hello world", "Hello world")
        example = sample_dataset[-1]
        kt = KeyTerm("hello", pipelines=sample_dataset.pipelines)
        trie = KeyTermTrie({kt}, normalized=False, add_capitalized=False)
        matches, _ = trie.find_in_tokens(example.ref.tokens)
        assert len(matches) == 0


class TestKeyTermMatch:
    """Tests for the structured KeyTermMatch returned by get_key_term_matches()."""

    def test_returns_key_term_match_instances(self, sample_dataset):
        """get_key_term_matches returns KeyTermMatch objects, not raw slices."""
        sample_dataset.add("the patient has diabetes", "the patient has diabetes")
        sample_dataset.add_vocabulary(Vocabulary(name="med").add_terms(["diabetes"]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="med")
        assert len(matches) == 1
        assert all(isinstance(m, KeyTermMatch) for m in matches)

    def test_span_and_token_slice(self, sample_dataset):
        """start/stop are token indices and token_slice indexes the parent TokenList."""
        sample_dataset.add("the quick brown fox", "the quick brown fox")
        sample_dataset.add_vocabulary(Vocabulary(name="phrases").add_terms(["quick brown"]))
        example = sample_dataset[-1]
        (match,) = example.ref.get_key_term_matches(vocab="phrases")
        assert (match.start, match.stop) == (1, 3)
        assert match.token_slice == slice(1, 3)
        assert example.ref.tokens[match.token_slice].raw == ["quick", "brown"]

    def test_references_and_derived_fields(self, sample_dataset):
        """text/key_term references and the derived tokens are correct."""
        sample_dataset.add("the quick brown fox", "the quick brown fox")
        sample_dataset.add_vocabulary(Vocabulary(name="phrases").add_terms(["quick brown"]))
        example = sample_dataset[-1]
        (match,) = example.ref.get_key_term_matches(vocab="phrases")
        assert match.text is example.ref
        assert isinstance(match.key_term, KeyTerm)
        assert match.key_term.raw == "quick brown"
        assert match.tokens.raw == ["quick", "brown"]

    def test_side_reflects_ref_vs_hyp(self, sample_dataset):
        """side mirrors the parent text's TextType for both ref and hyp matches."""
        sample_dataset.add("the brown fox", "the brown dog")
        sample_dataset.add_vocabulary(Vocabulary(name="animals").add_terms(["brown"]))
        example = sample_dataset[-1]
        (ref_match,) = example.ref.get_key_term_matches(vocab="animals")
        (hyp_match,) = example.hyp.get_key_term_matches(vocab="animals")
        assert ref_match.side == TextType.REF
        assert hyp_match.side == TextType.HYP

    def test_matched_tokens_preserve_original_casing(self, sample_dataset):
        """The matched tokens are the raw text even when matching is case-insensitive."""
        sample_dataset.add("Hello World", "hello world")
        sample_dataset.add_vocabulary(Vocabulary(name="greetings").add_terms(["hello"]))
        example = sample_dataset[-1]
        (match,) = example.ref.get_key_term_matches(vocab="greetings")
        assert match.tokens.raw == ["Hello"]
        assert match.key_term.raw == "hello"

    def test_subset_removal_keeps_richer_object(self, sample_dataset):
        """With allow_subset_matches=False the surviving longer match is a KeyTermMatch."""
        sample_dataset.add("the quick brown fox", "the quick brown fox")
        sample_dataset.add_vocabulary(Vocabulary(name="phrases").add_terms(["quick", "quick brown"]))
        example = sample_dataset[-1]
        (match,) = example.ref.get_key_term_matches(vocab="phrases", allow_subset_matches=False)
        assert isinstance(match, KeyTermMatch)
        assert match.tokens.raw == ["quick", "brown"]

    def test_colliding_key_terms_resolve_deterministically(self, sample_dataset):
        """When distinct raw terms collapse to the same token pattern, the reported key term
        is deterministic: the lexicographically smallest raw wins (here "Diabetes" < "diabetes")."""
        sample_dataset.add("the patient has diabetes", "the patient has diabetes")
        sample_dataset.add_vocabulary(Vocabulary(name="med").add_terms(["diabetes", "Diabetes"]))
        example = sample_dataset[-1]
        (match,) = example.ref.get_key_term_matches(vocab="med")
        assert match.key_term.raw == "Diabetes"
