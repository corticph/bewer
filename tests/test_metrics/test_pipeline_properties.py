"""Tests for pipeline property accessors on Metric and ExampleMetric."""

import pytest

from bewer import Dataset, Vocabulary, get_config
from bewer.flags import DEFAULT


class TestMetricPipelineProperties:
    """Tests for standardizer, tokenizer, normalizer properties on Metric."""

    def test_default_standardizer(self, sample_dataset):
        """Test that standardizer returns DEFAULT when not overridden."""
        wer = sample_dataset.metrics.wer()
        assert wer.standardizer == DEFAULT

    def test_default_tokenizer(self, sample_dataset):
        """Test that tokenizer returns DEFAULT when not overridden."""
        wer = sample_dataset.metrics.wer()
        assert wer.tokenizer == DEFAULT

    def test_default_normalizer(self, sample_dataset):
        """Test that normalizer returns DEFAULT when not overridden."""
        wer = sample_dataset.metrics.wer()
        assert wer.normalizer == DEFAULT

    def test_pipeline_matches_individual_properties(self, sample_dataset):
        """Test that pipeline tuple matches individual property values."""
        wer = sample_dataset.metrics.wer()
        assert wer.pipeline == (wer.standardizer, wer.tokenizer, wer.normalizer)

    def test_custom_tokenizer(self, sample_dataset):
        """Test that a non-default tokenizer is returned correctly."""
        wer = sample_dataset.metrics.wer(tokenizer="with_punctuation")
        assert wer.tokenizer == "with_punctuation"
        assert wer.standardizer == DEFAULT
        assert wer.normalizer == DEFAULT

    def test_custom_normalizer(self, sample_dataset):
        """Test that a non-default normalizer is returned correctly."""
        wer = sample_dataset.metrics.wer(normalizer="cased")
        assert wer.normalizer == "cased"


class TestExampleMetricPipelineProperties:
    """Tests for standardizer, tokenizer, normalizer properties on ExampleMetric."""

    def test_delegates_to_parent_standardizer(self, sample_dataset):
        """Test that ExampleMetric.standardizer delegates to parent metric."""
        example = sample_dataset[0]
        wer_example = example.metrics.wer()
        wer_dataset = sample_dataset.metrics.wer()
        assert wer_example.standardizer == wer_dataset.standardizer

    def test_delegates_to_parent_tokenizer(self, sample_dataset):
        """Test that ExampleMetric.tokenizer delegates to parent metric."""
        example = sample_dataset[0]
        wer_example = example.metrics.wer()
        wer_dataset = sample_dataset.metrics.wer()
        assert wer_example.tokenizer == wer_dataset.tokenizer

    def test_delegates_to_parent_normalizer(self, sample_dataset):
        """Test that ExampleMetric.normalizer delegates to parent metric."""
        example = sample_dataset[0]
        wer_example = example.metrics.wer()
        wer_dataset = sample_dataset.metrics.wer()
        assert wer_example.normalizer == wer_dataset.normalizer

    def test_delegates_to_parent_pipeline(self, sample_dataset):
        """Test that ExampleMetric.pipeline delegates to parent metric."""
        example = sample_dataset[0]
        wer_example = example.metrics.wer()
        wer_dataset = sample_dataset.metrics.wer()
        assert wer_example.pipeline == wer_dataset.pipeline

    def test_custom_pipeline_propagates(self, sample_dataset):
        """Test that custom pipeline settings propagate to ExampleMetric."""
        example = sample_dataset[0]
        wer_example = example.metrics.wer(tokenizer="with_punctuation", normalizer="cased")
        assert wer_example.standardizer == DEFAULT
        assert wer_example.tokenizer == "with_punctuation"
        assert wer_example.normalizer == "cased"


class TestNoneStages:
    """Tests for skipping a preprocessing stage with the reserved ``none`` name."""

    def test_none_normalizer_compares_standardized_tokens(self, empty_dataset):
        """normalizer="none" skips token normalization, so case differences count as edits."""
        empty_dataset.add("Hello world", "hello world")
        assert empty_dataset.metrics.wer().value == 0.0
        assert empty_dataset.metrics.wer(normalizer="none").value == 0.5
        assert empty_dataset[0].ref.tokens.standardized == ["Hello", "world"]

    def test_none_standardizer_keeps_submitted_text(self, empty_dataset):
        """standardizer="none" skips apostrophe unification, so variant apostrophes differ."""
        empty_dataset.add("it\u2019s fine", "it's fine")
        assert empty_dataset.metrics.wer(normalizer="none").value == 0.0
        assert empty_dataset.metrics.wer(standardizer="none", normalizer="none").value == 0.5

    def test_none_stages_apply_to_key_term_matching(self):
        """Key-term matching follows the selected normalizer."""
        config = get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["Fox"]))
        dataset = Dataset(config=config)
        dataset.add("the fox jumps", "the fox jumps")
        assert dataset.metrics.ktr(vocab="animals").num_ref_terms == 1
        assert dataset.metrics.ktr(vocab="animals", normalizer="none").num_ref_terms == 0

    def test_none_tokenizer_is_rejected(self, sample_dataset):
        """Tokenization cannot be skipped; the request fails before the dataset freezes."""
        with pytest.raises(ValueError, match="tokenization cannot be skipped"):
            sample_dataset.metrics.wer(tokenizer="none")
        assert not sample_dataset.is_frozen

    @pytest.mark.parametrize("metric", ["wer", "insertion_rate"])
    def test_normalized_param_is_rejected_with_migration_hint(self, sample_dataset, metric):
        """The removed ``normalized`` parameter points to ``normalizer="none"``."""
        with pytest.raises(ValueError, match="normalizer='none'"):
            sample_dataset.metrics.get(metric)(normalized=False)
        assert not sample_dataset.is_frozen
