"""Tests for bewer.core.dataset module."""

import os
import tempfile

import pandas as pd
import pytest

from bewer.core.dataset import Dataset, TextList, TextTokenList
from bewer.core.example import Example


class TestDatasetInit:
    """Tests for Dataset.__init__()."""

    def test_default_config(self):
        """Test Dataset initialization with default config."""
        dataset = Dataset()
        assert dataset.config is not None
        assert dataset.pipelines is not None

    def test_empty_examples_list(self):
        """Test that examples list is initially empty."""
        dataset = Dataset()
        assert dataset.examples == []
        assert len(dataset) == 0

    def test_metrics_collection_created(self):
        """Test that metrics collection is created."""
        dataset = Dataset()
        assert dataset.metrics is not None


class TestDatasetAdd:
    """Tests for Dataset.add() method."""

    def test_add_single_example(self, empty_dataset):
        """Test adding a single example."""
        empty_dataset.add("hello", "hi")
        assert len(empty_dataset) == 1

    def test_add_multiple_examples(self, empty_dataset):
        """Test adding multiple examples."""
        empty_dataset.add("one", "1")
        empty_dataset.add("two", "2")
        empty_dataset.add("three", "3")
        assert len(empty_dataset) == 3

    def test_added_example_is_example_object(self, empty_dataset):
        """Test that added item is an Example object."""
        empty_dataset.add("hello", "hi")
        assert isinstance(empty_dataset[0], Example)

    def test_added_example_has_correct_text(self, empty_dataset):
        """Test that added example has correct ref/hyp."""
        empty_dataset.add("reference text", "hypothesis text")
        assert empty_dataset[0].ref.raw == "reference text"
        assert empty_dataset[0].hyp.raw == "hypothesis text"

    def test_added_example_has_correct_index(self, empty_dataset):
        """Test that added examples have correct indices."""
        empty_dataset.add("one", "1")
        empty_dataset.add("two", "2")
        assert empty_dataset[0].index == 0
        assert empty_dataset[1].index == 1

    def test_add_with_keywords(self, empty_dataset):
        """Test adding example with keywords."""
        empty_dataset.add("the quick brown fox", "the quick brown dog", keywords={"animals": ["fox"]})
        assert "animals" in empty_dataset[0].keywords


class TestDatasetLoadPandas:
    """Tests for Dataset.load_pandas() method."""

    def test_load_basic_dataframe(self, empty_dataset):
        """Test loading a basic DataFrame."""
        df = pd.DataFrame({"ref": ["hello world", "test phrase"], "hyp": ["hello world", "test sentence"]})
        empty_dataset.load_pandas(df)
        assert len(empty_dataset) == 2

    def test_load_custom_column_names(self, empty_dataset):
        """Test loading DataFrame with custom column names."""
        df = pd.DataFrame({"reference": ["hello", "world"], "hypothesis": ["hi", "earth"]})
        empty_dataset.load_pandas(df, ref_col="reference", hyp_col="hypothesis")
        assert len(empty_dataset) == 2
        assert empty_dataset[0].ref.raw == "hello"
        assert empty_dataset[0].hyp.raw == "hi"

    def test_load_invalid_type_raises(self, empty_dataset):
        """Test that loading non-DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            empty_dataset.load_pandas({"ref": ["hello"], "hyp": ["hi"]})


class TestDatasetLoadCsv:
    """Tests for Dataset.load_csv() method."""

    def test_load_csv_file(self, empty_dataset):
        """Test loading from CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ref,hyp\n")
            f.write("hello world,hello world\n")
            f.write("test phrase,test sentence\n")
            csv_path = f.name

        try:
            empty_dataset.load_csv(csv_path)
            assert len(empty_dataset) == 2
        finally:
            os.unlink(csv_path)

    def test_load_csv_custom_columns(self, empty_dataset):
        """Test loading CSV with custom column names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("reference,hypothesis\n")
            f.write("hello,hi\n")
            csv_path = f.name

        try:
            empty_dataset.load_csv(csv_path, ref_col="reference", hyp_col="hypothesis")
            assert len(empty_dataset) == 1
            assert empty_dataset[0].ref.raw == "hello"
        finally:
            os.unlink(csv_path)


class TestDatasetRefsHyps:
    """Tests for Dataset.refs and Dataset.hyps properties."""

    def test_refs_returns_textlist(self, sample_dataset):
        """Test that refs property returns TextList."""
        refs = sample_dataset.refs
        assert isinstance(refs, TextList)

    def test_refs_correct_length(self, sample_dataset):
        """Test that refs has correct length."""
        refs = sample_dataset.refs
        assert len(refs) == len(sample_dataset)

    def test_hyps_returns_textlist(self, sample_dataset):
        """Test that hyps property returns TextList."""
        hyps = sample_dataset.hyps
        assert isinstance(hyps, TextList)

    def test_hyps_correct_length(self, sample_dataset):
        """Test that hyps has correct length."""
        hyps = sample_dataset.hyps
        assert len(hyps) == len(sample_dataset)


class TestDatasetContainerProtocol:
    """Tests for Dataset container protocol (__len__, __getitem__, __iter__)."""

    def test_len(self, sample_dataset):
        """Test __len__ returns correct count."""
        assert len(sample_dataset) == 3

    def test_getitem(self, sample_dataset):
        """Test __getitem__ returns correct example."""
        example = sample_dataset[0]
        assert isinstance(example, Example)
        assert example.ref.raw == "hello world"

    def test_getitem_negative_index(self, sample_dataset):
        """Test __getitem__ with negative index."""
        example = sample_dataset[-1]
        assert example.ref.raw == "testing one two three"

    def test_iter(self, sample_dataset):
        """Test __iter__ iterates over examples."""
        examples = list(sample_dataset)
        assert len(examples) == 3
        assert all(isinstance(e, Example) for e in examples)


class TestDatasetRepr:
    """Tests for Dataset.__repr__()."""

    def test_repr(self, sample_dataset):
        """Test string representation."""
        repr_str = repr(sample_dataset)
        assert "Dataset" in repr_str
        assert "3" in repr_str  # number of examples


class TestTextList:
    """Tests for TextList class."""

    def test_raw_property(self, sample_dataset):
        """Test raw property returns list of strings."""
        refs = sample_dataset.refs
        raw = refs.raw
        assert isinstance(raw, list)
        assert all(isinstance(r, str) for r in raw)

    def test_standardized_property(self, sample_dataset):
        """Test standardized property returns list of strings."""
        refs = sample_dataset.refs
        standardized = refs.standardized
        assert isinstance(standardized, list)
        assert all(isinstance(s, str) for s in standardized)

    def test_tokens_property(self, sample_dataset):
        """Test tokens property returns TextTokenList."""
        refs = sample_dataset.refs
        tokens = refs.tokens
        assert isinstance(tokens, TextTokenList)

    def test_slice_returns_textlist(self, sample_dataset):
        """Test that slicing returns TextList."""
        refs = sample_dataset.refs
        sliced = refs[0:2]
        assert isinstance(sliced, TextList)
        assert len(sliced) == 2

    def test_add_returns_textlist(self, sample_dataset):
        """Test that adding TextLists returns TextList."""
        refs = sample_dataset.refs
        hyps = sample_dataset.hyps
        combined = refs + hyps
        assert isinstance(combined, TextList)
        assert len(combined) == len(refs) + len(hyps)


class TestTextTokenList:
    """Tests for TextTokenList class."""

    def test_raw_property(self, sample_dataset):
        """Test raw property returns nested list."""
        tokens = sample_dataset.refs.tokens
        raw = tokens.raw
        assert isinstance(raw, list)
        assert isinstance(raw[0], list)

    def test_normalized_property(self, sample_dataset):
        """Test normalized property returns nested list."""
        tokens = sample_dataset.refs.tokens
        normalized = tokens.normalized
        assert isinstance(normalized, list)
        assert isinstance(normalized[0], list)

    def test_flat_property(self, sample_dataset):
        """Test flat property returns TokenList."""
        tokens = sample_dataset.refs.tokens
        flat = tokens.flat
        from bewer.core.text import TokenList

        assert isinstance(flat, TokenList)
