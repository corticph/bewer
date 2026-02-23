"""Tests for pipeline property accessors on Metric and ExampleMetric."""

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

    def test_custom_standardizer(self, sample_dataset):
        """Test that a custom standardizer is returned correctly."""
        wer = sample_dataset.metrics.wer(standardizer="custom_std")
        assert wer.standardizer == "custom_std"
        assert wer.tokenizer == DEFAULT
        assert wer.normalizer == DEFAULT

    def test_custom_tokenizer(self, sample_dataset):
        """Test that a custom tokenizer is returned correctly."""
        wer = sample_dataset.metrics.wer(tokenizer="custom_tok")
        assert wer.tokenizer == "custom_tok"

    def test_custom_normalizer(self, sample_dataset):
        """Test that a custom normalizer is returned correctly."""
        wer = sample_dataset.metrics.wer(normalizer="custom_norm")
        assert wer.normalizer == "custom_norm"


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
        wer_example = example.metrics.wer(standardizer="custom_std", tokenizer="custom_tok")
        assert wer_example.standardizer == "custom_std"
        assert wer_example.tokenizer == "custom_tok"
        assert wer_example.normalizer == DEFAULT
