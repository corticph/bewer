"""Tests for bewer.core.example module."""

import pytest

from bewer import Vocabulary
from bewer.core.example import Example
from bewer.core.text import Text, TextType


class TestExampleStandalone:
    """Tests for an Example constructed without a parent Dataset (src=None)."""

    def test_constructs_without_dataset(self, pipelines):
        """A standalone Example can be built given only a Pipelines registry."""
        example = Example("a b c", "a x c", pipelines=pipelines)
        assert example.src is None
        assert example.ref.tokens.raw == ["a", "b", "c"]

    def test_get_key_term_matches_returns_empty(self, pipelines):
        """Key term matching needs the dataset trie; without a Dataset it yields no matches."""
        example = Example("a b c", "a x c", pipelines=pipelines)
        assert example.ref.get_key_term_matches("v") == []

    def test_vocabs_empty_without_dataset(self, pipelines):
        """vocabs is empty for an Example with no parent Dataset."""
        example = Example("a b c", "a x c", pipelines=pipelines)
        assert example.vocabs == set()

    def test_metrics_raise_clear_error(self, pipelines):
        """Dataset-backed metrics are unavailable for a parentless Example."""
        example = Example("a b c", "a x c", pipelines=pipelines)
        with pytest.raises(ValueError, match="without a parent Dataset"):
            example.metrics.wer()


class TestExampleInit:
    """Tests for Example.__init__()."""

    def test_creates_ref_text_object(self, sample_example):
        """Test that Example creates ref as Text object."""
        assert isinstance(sample_example.ref, Text)
        assert sample_example.ref.raw == "hello world"

    def test_creates_hyp_text_object(self, sample_example):
        """Test that Example creates hyp as Text object."""
        assert isinstance(sample_example.hyp, Text)
        assert sample_example.hyp.raw == "hello world"

    def test_ref_text_type(self, sample_example):
        """Test that ref has correct TextType."""
        assert sample_example.ref.text_type == TextType.REF

    def test_hyp_text_type(self, sample_example):
        """Test that hyp has correct TextType."""
        assert sample_example.hyp.text_type == TextType.HYP

    def test_index_property(self, sample_dataset):
        """Test that examples have correct indices."""
        assert sample_dataset[0].index == 0
        assert sample_dataset[1].index == 1
        assert sample_dataset[2].index == 2

    def test_src_property(self, sample_example):
        """Test that src property returns the dataset."""
        assert sample_example.src is not None


class TestExamplePrepareAndValidateKeyTerms:
    """Tests for Example key term preparation and validation."""

    def test_key_term_not_in_ref_no_matches(self, sample_dataset):
        """Test that key term not in reference produces no matches."""
        sample_dataset.add("hello world", "hello world")
        sample_dataset.add_vocabulary(Vocabulary(name="missing").add_terms(["nonexistent"]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="missing")
        assert len(matches) == 0

    def test_case_insensitive_key_term_matching(self, sample_dataset):
        """Test that key term matching is case insensitive."""
        sample_dataset.add("Hello World", "hello world")
        sample_dataset.add_vocabulary(Vocabulary(name="greetings").add_terms(["hello"]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="greetings")
        assert len(matches) == 1
        assert matches[0].tokens.raw == ["Hello"]

    def test_empty_key_term_list_resolves_without_error(self, sample_dataset):
        """Test that an empty key term list does not cause key term match resolution to fail."""
        sample_dataset.add("hello world", "hello world")
        sample_dataset.add_vocabulary(Vocabulary(name="greetings").add_terms([]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="greetings")
        assert matches == []


class TestExampleVocabs:
    """Tests for Example.vocabs property."""

    def test_vocabs_empty_when_no_key_terms(self, sample_dataset):
        """Test that vocabs is empty when no key terms are set."""
        example = sample_dataset[0]
        assert example.vocabs == set()

    def test_vocabs_includes_global_dataset_vocabs(self, sample_dataset):
        """Test that vocabs includes global key term vocabularies from the parent dataset."""
        sample_dataset.add_vocabulary(Vocabulary(name="global_terms").add_terms(["hello"]))
        example = sample_dataset[0]
        assert "global_terms" in example.vocabs


class TestExampleRepr:
    """Tests for Example.__repr__()."""

    def test_repr_short_texts(self, sample_example):
        """Test repr with short texts."""
        repr_str = repr(sample_example)
        assert "Example" in repr_str
        assert "ref=" in repr_str
        assert "hyp=" in repr_str
        assert "hello world" in repr_str

    def test_repr_long_texts(self, sample_dataset):
        """Test repr truncates long texts."""
        long_text = "a" * 100
        sample_dataset.add(long_text, long_text)
        example = sample_dataset[-1]
        repr_str = repr(example)
        assert "..." in repr_str


class TestExampleHash:
    """Tests for Example.__hash__()."""

    def test_hash_includes_ref_and_hyp(self, sample_dataset):
        """Test that hash is based on ref, hyp, and index."""
        example1 = sample_dataset[0]
        example2 = sample_dataset[0]
        assert hash(example1) == hash(example2)

    def test_different_examples_different_hash(self, sample_dataset):
        """Test that different examples have different hashes."""
        example1 = sample_dataset[0]
        example2 = sample_dataset[1]
        assert hash(example1) != hash(example2)


class TestTextGetKeyTermMatches:
    """Tests for Text.get_key_term_matches() using global and local key terms."""

    def test_global_key_terms_both_matched(self, sample_dataset):
        """Global vocab (from add_vocabulary) produces all matches by default."""
        sample_dataset.add("the quick brown fox", "the quick brown dog")
        sample_dataset.add_vocabulary(Vocabulary(name="animals").add_terms(["fox", "brown"]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="animals")
        matched_raws = sorted(m.tokens.raw for m in matches)
        assert ["brown"] in matched_raws
        assert ["fox"] in matched_raws

    def test_global_key_terms_match_count(self, sample_dataset):
        """Global key terms produce the expected number of matches."""
        sample_dataset.add("the quick brown fox", "the quick brown dog")
        sample_dataset.add_vocabulary(Vocabulary(name="animals").add_terms(["fox"]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="animals")
        assert len(matches) == 1

    def test_allow_subset_matches_true_deduplicates_exact(self, sample_dataset):
        """With allow_subset_matches=True, exact duplicate matches from global vocab are deduplicated."""
        sample_dataset.add("the quick brown fox", "the quick brown dog")
        sample_dataset.add_vocabulary(Vocabulary(name="animals").add_terms(["fox"]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="animals", allow_subset_matches=True)
        assert len(matches) == 1

    def test_allow_subset_matches_false_deduplicates(self, sample_dataset):
        """With allow_subset_matches=False, subset matches from global vocab are deduplicated."""
        sample_dataset.add("the quick brown fox", "the quick brown dog")
        sample_dataset.add_vocabulary(Vocabulary(name="animals").add_terms(["fox"]))
        example = sample_dataset[-1]
        matches = example.ref.get_key_term_matches(vocab="animals", allow_subset_matches=False)
        assert len(matches) == 1


class TestExampleMetrics:
    """Tests for Example.metrics attribute."""

    def test_metrics_collection_exists(self, sample_example):
        """Test that metrics collection is created."""
        assert hasattr(sample_example, "metrics")
        assert sample_example.metrics is not None
