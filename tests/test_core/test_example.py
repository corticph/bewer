"""Tests for bewer.core.example module."""

import warnings

from bewer.core.keyword import KeywordNotFoundWarning
from bewer.core.text import Text, TextType


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


class TestExamplePrepareAndValidateKeywords:
    """Tests for Example keyword preparation and validation."""

    def test_none_keywords_returns_empty_dict(self, sample_dataset):
        """Test that None keywords returns empty dict."""
        example = sample_dataset[0]
        assert example.keywords == {}

    def test_valid_keywords_converted(self, sample_dataset):
        """Test that valid keywords are converted to Text objects."""
        sample_dataset.add("the quick brown fox", "the quick brown dog", keywords={"animals": ["fox"]})
        example = sample_dataset[-1]
        assert "animals" in example.keywords
        assert len(example.keywords["animals"]) == 1
        assert isinstance(example.keywords["animals"].pop(), Text)

    def test_keyword_not_in_ref_warns(self, sample_dataset):
        """Test that keyword not in reference issues warning when trie matches are computed."""
        sample_dataset.add("hello world", "hello world", keywords={"missing": ["nonexistent"]})
        example = sample_dataset[-1]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            example.get_keyword_matches(vocab="missing")
            assert len([x for x in w if issubclass(x.category, KeywordNotFoundWarning)]) > 0

    def test_keyword_not_in_ref_no_matches(self, sample_dataset):
        """Test that keyword not in reference produces no matches."""
        sample_dataset.add("hello world", "hello world", keywords={"missing": ["nonexistent"]})
        example = sample_dataset[-1]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matches = example.get_keyword_matches(vocab="missing")
        assert len(matches) == 0

    def test_case_insensitive_keyword_matching(self, sample_dataset):
        """Test that keyword matching is case insensitive."""
        sample_dataset.add("Hello World", "hello world", keywords={"greetings": ["hello"]})
        example = sample_dataset[-1]
        assert "greetings" in example.keywords
        assert len(example.keywords["greetings"]) == 1

    def test_multiple_keyword_groups(self, sample_dataset):
        """Test multiple keyword groups."""
        sample_dataset.add(
            "the quick brown fox jumps",
            "the quick brown dog jumps",
            keywords={"colors": ["brown"], "animals": ["fox"], "actions": ["jumps"]},
        )
        example = sample_dataset[-1]
        assert len(example.keywords) == 3
        assert "colors" in example.keywords
        assert "animals" in example.keywords
        assert "actions" in example.keywords


class TestExampleVocabs:
    """Tests for Example.vocabs property."""

    def test_vocabs_empty_when_no_keywords(self, sample_dataset):
        """Test that vocabs is empty when no keywords are set."""
        example = sample_dataset[0]
        assert example.vocabs == set()

    def test_vocabs_from_example_keywords(self, sample_dataset):
        """Test that vocabs includes example-level keyword vocabularies."""
        sample_dataset.add(
            "the quick brown fox",
            "the quick brown dog",
            keywords={"animals": ["fox"], "colors": ["brown"]},
        )
        example = sample_dataset[-1]
        assert example.vocabs == {"animals", "colors"}

    def test_vocabs_includes_dynamic_dataset_vocabs(self, sample_dataset):
        """Test that vocabs includes dynamic keyword vocabularies from the parent dataset."""
        sample_dataset._dynamic_keyword_vocabs["global_terms"] = set()
        example = sample_dataset[0]
        assert "global_terms" in example.vocabs

    def test_vocabs_merges_example_and_dataset_vocabs(self, sample_dataset):
        """Test that vocabs merges both example-level and dataset-level vocabularies."""
        sample_dataset.add(
            "hello world",
            "hello world",
            keywords={"greetings": ["hello"]},
        )
        sample_dataset._dynamic_keyword_vocabs["global_terms"] = set()
        example = sample_dataset[-1]
        assert example.vocabs == {"greetings", "global_terms"}


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


class TestExampleGetKeywordMatches:
    """Tests for Example.get_keyword_matches() merging example and dataset keywords."""

    def test_merges_example_and_dataset_keywords(self, sample_dataset):
        """Matches from both example-level and dataset-level keywords are returned."""
        sample_dataset.add(
            "the quick brown fox",
            "the quick brown dog",
            keywords={"animals": ["fox"]},
        )
        sample_dataset.add_keyword_list("animals", ["brown"])
        example = sample_dataset[-1]
        matches = example.get_keyword_matches(vocab="animals")
        matched_raws = sorted(example.ref.tokens[m].raw for m in matches)
        assert ["brown"] in matched_raws
        assert ["fox"] in matched_raws

    def test_dataset_only_keywords_no_warning(self, sample_dataset):
        """Dataset-level keywords produce matches without KeywordNotFoundWarning."""
        sample_dataset.add("the quick brown fox", "the quick brown dog")
        sample_dataset.add_keyword_list("animals", ["fox"])
        example = sample_dataset[-1]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            matches = example.get_keyword_matches(vocab="animals")
        assert len(matches) == 1
        assert len([x for x in w if issubclass(x.category, KeywordNotFoundWarning)]) == 0

    def test_cached_no_duplicate_warnings(self, sample_dataset):
        """Second call returns cached result and does not re-emit warnings."""
        sample_dataset.add("hello world", "hello world", keywords={"missing": ["nonexistent"]})
        example = sample_dataset[-1]
        # First call triggers the warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            example.get_keyword_matches(vocab="missing")
        # Second call should be cached — no new warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            example.get_keyword_matches(vocab="missing")
        assert len([x for x in w if issubclass(x.category, KeywordNotFoundWarning)]) == 0

    def test_allow_subsets_true_deduplicates_exact(self, sample_dataset):
        """With allow_subsets=True, exact duplicate matches from both tries are deduplicated."""
        sample_dataset.add(
            "the quick brown fox",
            "the quick brown dog",
            keywords={"animals": ["fox"]},
        )
        sample_dataset.add_keyword_list("animals", ["fox"])
        example = sample_dataset[-1]
        matches = example.get_keyword_matches(vocab="animals", allow_subsets=True)
        assert len(matches) == 1

    def test_allow_subsets_false_deduplicates(self, sample_dataset):
        """With allow_subsets=False, duplicate matches from both tries are deduplicated."""
        sample_dataset.add(
            "the quick brown fox",
            "the quick brown dog",
            keywords={"animals": ["fox"]},
        )
        sample_dataset.add_keyword_list("animals", ["fox"])
        example = sample_dataset[-1]
        matches = example.get_keyword_matches(vocab="animals", allow_subsets=False)
        assert len(matches) == 1


class TestExampleMetrics:
    """Tests for Example.metrics attribute."""

    def test_metrics_collection_exists(self, sample_example):
        """Test that metrics collection is created."""
        assert hasattr(sample_example, "metrics")
        assert sample_example.metrics is not None
