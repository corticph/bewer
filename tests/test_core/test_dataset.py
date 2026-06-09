"""Tests for bewer.core.dataset module."""

import os
import tempfile

import pandas as pd
import pytest

from bewer.core.dataset import Dataset, DatasetFrozenError, TextList, TextTokenList
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

    def test_add_with_key_terms(self, empty_dataset):
        """Test adding example with key terms."""
        empty_dataset.add("the quick brown fox", "the quick brown dog", key_terms={"animals": ["fox"]})
        assert "animals" in empty_dataset[0].key_terms


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


class TestDatasetLoadJsonl:
    """Tests for Dataset.load_jsonl() method."""

    def test_load_jsonl_file(self, empty_dataset):
        """Test loading from JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"ref": "hello world", "hyp": "hello world"}\n')
            f.write('{"ref": "test phrase", "hyp": "test sentence"}\n')
            jsonl_path = f.name

        try:
            empty_dataset.load_jsonl(jsonl_path)
            assert len(empty_dataset) == 2
            assert empty_dataset[0].ref.raw == "hello world"
            assert empty_dataset[1].hyp.raw == "test sentence"
        finally:
            os.unlink(jsonl_path)

    def test_load_jsonl_custom_columns(self, empty_dataset):
        """Test loading JSONL with custom column names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"reference": "hello", "hypothesis": "hi"}\n')
            jsonl_path = f.name

        try:
            empty_dataset.load_jsonl(jsonl_path, ref_col="reference", hyp_col="hypothesis")
            assert len(empty_dataset) == 1
            assert empty_dataset[0].ref.raw == "hello"
            assert empty_dataset[0].hyp.raw == "hi"
        finally:
            os.unlink(jsonl_path)

    def test_load_jsonl_with_key_term_cols(self, empty_dataset):
        """Test loading JSONL with key term columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"ref": "the quick brown fox", "hyp": "the quick brown dog", "animals": ["fox"]}\n')
            jsonl_path = f.name

        try:
            empty_dataset.load_jsonl(jsonl_path, key_term_cols=["animals"])
            assert len(empty_dataset) == 1
            assert "animals" in empty_dataset[0].key_terms
        finally:
            os.unlink(jsonl_path)


class TestDatasetAddKeyTermFile:
    """Tests for Dataset.add_key_term_file() method."""

    def test_add_key_term_file(self, empty_dataset):
        """Test loading key terms from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("fox\nbrown\n")
            key_term_path = f.name

        try:
            empty_dataset.add_key_term_file("animals", key_term_path)
            assert "animals" in empty_dataset._global_key_term_vocabs
            kt_raws = {kt.raw for kt in empty_dataset._global_key_term_vocabs["animals"]}
            assert "fox" in kt_raws
            assert "brown" in kt_raws
        finally:
            os.unlink(key_term_path)

    def test_add_key_term_file_not_found(self, empty_dataset):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            empty_dataset.add_key_term_file("animals", "/nonexistent/path/key_terms.txt")

    def test_add_key_term_file_matches_existing_examples(self, empty_dataset):
        """Test that key terms from file are matched against existing examples."""
        empty_dataset.add("the quick brown fox", "the quick brown dog")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("fox\n")
            key_term_path = f.name

        try:
            empty_dataset.add_key_term_file("animals", key_term_path)
            assert "animals" in empty_dataset._global_key_term_vocabs
            matches = empty_dataset[0].ref.get_key_term_matches(vocab="animals")
            assert len(matches) == 1
        finally:
            os.unlink(key_term_path)


class TestDatasetAddKeyTermListValidation:
    """Tests for input validation in Dataset.add_key_term_list()."""

    def test_add_key_term_list_string_raises(self, empty_dataset):
        """Test that passing a string raises TypeError."""
        with pytest.raises(TypeError, match="must be an iterable"):
            empty_dataset.add_key_term_list("test", "not_a_list")

    def test_add_key_term_list_non_iterable_raises(self, empty_dataset):
        """Test that passing a non-iterable raises TypeError."""
        with pytest.raises(TypeError, match="must be an iterable"):
            empty_dataset.add_key_term_list("test", 42)


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


class TestDatasetLanguage:
    """Tests for Dataset language parameter."""

    def test_default_language_strips_danish_chars(self):
        """Test that default dataset strips language-specific chars."""
        ds = Dataset()
        ds.add("patienten har høj blodtryk", "patienten har hoj blodtryk")
        normalized = ds[0].ref.tokens.normalized
        assert "høj" not in normalized
        assert "hoj" in normalized

    def test_danish_language_preserves_chars(self):
        """Test that language='da' retains æ/ø/å in normalization."""
        ds = Dataset(language="da")
        ds.add("patienten har høj blodtryk", "patienten har hoj blodtryk")
        normalized = ds[0].ref.tokens.normalized
        assert "høj" in normalized

    def test_german_language_preserves_chars(self):
        """Test that language='de' retains ä/ö/ü/ß in normalization."""
        ds = Dataset(language="de")
        ds.add("straße", "strasse")
        normalized = ds[0].ref.tokens.normalized
        assert "straße" in normalized

    def test_french_language_preserves_chars(self):
        """Test that language='fr' retains accented chars in normalization."""
        ds = Dataset(language="fr")
        ds.add("café", "cafe")
        normalized = ds[0].ref.tokens.normalized
        assert "café" in normalized

    def test_english_language_same_as_default(self):
        """Test that language='en' behaves the same as no language arg."""
        ds_default = Dataset()
        ds_en = Dataset(language="en")
        ds_default.add("café", "cafe")
        ds_en.add("café", "cafe")
        assert ds_default[0].ref.tokens.normalized == ds_en[0].ref.tokens.normalized

    def test_unknown_language_raises(self):
        """Test that an unknown language raises ValueError."""
        with pytest.raises(ValueError, match="Unknown language"):
            Dataset(language="xx")

    def test_language_does_not_affect_non_language_pipeline(self):
        """Test that language overlay only changes normalizer, not other pipeline steps."""
        ds_default = Dataset()
        ds_da = Dataset(language="da")
        ds_default.add("Hello World", "hello world")
        ds_da.add("Hello World", "hello world")
        # Standardization should be the same
        assert ds_default[0].ref.standardized == ds_da[0].ref.standardized


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


class TestDatasetFreeze:
    """Tests for the freeze lifecycle (frozen property, auto-freeze, mutation guards)."""

    def test_new_dataset_not_frozen(self, empty_dataset):
        """A freshly created dataset is not frozen."""
        assert empty_dataset.frozen is False

    def test_requesting_metric_freezes(self, sample_dataset):
        """Requesting a metric freezes the dataset."""
        assert sample_dataset.frozen is False
        sample_dataset.metrics.wer()
        assert sample_dataset.frozen is True

    def test_accessing_metric_factory_does_not_freeze(self, sample_dataset):
        """Referencing the metric factory without calling it does not freeze."""
        _ = sample_dataset.metrics.wer  # bound factory, not called
        assert sample_dataset.frozen is False

    def test_list_metrics_does_not_freeze(self, sample_dataset):
        """Listing metrics does not freeze the dataset."""
        sample_dataset.metrics.list_metrics()
        assert sample_dataset.frozen is False

    def test_example_metric_request_freezes(self, sample_dataset):
        """Requesting an example-level metric also freezes the dataset."""
        sample_dataset[0].metrics.wer()
        assert sample_dataset.frozen is True

    def test_manual_freeze(self, empty_dataset):
        """freeze() sets the frozen flag and is idempotent."""
        empty_dataset.freeze()
        assert empty_dataset.frozen is True
        empty_dataset.freeze()  # idempotent, no error
        assert empty_dataset.frozen is True

    def test_building_before_freeze_works(self, empty_dataset):
        """Data can be added freely before the dataset is frozen."""
        empty_dataset.add("hello", "hi")
        empty_dataset.add_key_term_list("v", ["hello"])
        assert len(empty_dataset) == 1

    def test_add_after_freeze_raises(self, sample_dataset):
        """add() raises once the dataset is frozen."""
        sample_dataset.freeze()
        with pytest.raises(DatasetFrozenError, match="frozen Dataset"):
            sample_dataset.add("foo", "bar")

    def test_add_key_term_list_after_freeze_raises(self, sample_dataset):
        sample_dataset.freeze()
        with pytest.raises(DatasetFrozenError):
            sample_dataset.add_key_term_list("v", ["foo"])

    def test_load_pandas_after_freeze_raises(self, sample_dataset):
        sample_dataset.freeze()
        df = pd.DataFrame({"ref": ["a"], "hyp": ["b"]})
        with pytest.raises(DatasetFrozenError):
            sample_dataset.load_pandas(df)

    def test_load_csv_after_freeze_raises(self, sample_dataset, tmp_path):
        sample_dataset.freeze()
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("ref,hyp\nhello,hi\n")
        with pytest.raises(DatasetFrozenError):
            sample_dataset.load_csv(str(csv_path))

    def test_load_jsonl_after_freeze_raises(self, sample_dataset, tmp_path):
        sample_dataset.freeze()
        jsonl_path = tmp_path / "data.jsonl"
        jsonl_path.write_text('{"ref": "hello", "hyp": "hi"}\n')
        with pytest.raises(DatasetFrozenError):
            sample_dataset.load_jsonl(str(jsonl_path))

    def test_load_dataset_after_freeze_raises(self, sample_dataset):
        sample_dataset.freeze()
        with pytest.raises(DatasetFrozenError):
            sample_dataset.load_dataset(None)

    def test_add_key_term_file_after_freeze_raises(self, sample_dataset, tmp_path):
        sample_dataset.freeze()
        kt_path = tmp_path / "kt.txt"
        kt_path.write_text("fox\n")
        with pytest.raises(DatasetFrozenError):
            sample_dataset.add_key_term_file("v", str(kt_path))

    def test_add_after_metric_computation_raises(self, sample_dataset):
        """End-to-end: computing a metric value then adding data raises."""
        sample_dataset.metrics.wer().value
        with pytest.raises(DatasetFrozenError):
            sample_dataset.add("foo", "bar")


class TestDatasetClone:
    """Tests for Dataset.clone()."""

    def test_clone_copies_examples(self, sample_dataset):
        clone = sample_dataset.clone()
        assert clone.refs.raw == sample_dataset.refs.raw
        assert clone.hyps.raw == sample_dataset.hyps.raw

    def test_clone_is_unfrozen(self, sample_dataset):
        sample_dataset.freeze()
        clone = sample_dataset.clone()
        assert clone.frozen is False
        clone.add("foo", "bar")  # modifiable
        assert len(clone) == len(sample_dataset) + 1

    def test_clone_is_independent(self, empty_dataset):
        empty_dataset.add("hello", "hi")
        clone = empty_dataset.clone()
        clone.add("foo", "bar")
        assert len(empty_dataset) == 1
        assert len(clone) == 2

    def test_clone_copies_global_vocab(self, empty_dataset):
        empty_dataset.add("the quick brown fox", "the quick brown dog")
        empty_dataset.add_key_term_list("animals", ["fox"])
        clone = empty_dataset.clone()
        assert "animals" in clone._global_key_term_vocabs
        assert {kt.raw for kt in clone._global_key_term_vocabs["animals"]} == {"fox"}

    def test_clone_copies_local_key_terms(self, empty_dataset):
        empty_dataset.add("the quick brown fox", "the quick brown dog", key_terms={"animals": ["fox"]})
        clone = empty_dataset.clone()
        assert "animals" in clone[0].key_terms
        assert {kt.raw for kt in clone[0].key_terms["animals"]} == {"fox"}

    def test_clone_metric_values_match(self, dataset_with_errors):
        original_value = dataset_with_errors.metrics.wer().value
        clone = dataset_with_errors.clone()
        assert clone.metrics.wer().value == original_value

    def test_clone_then_extend_and_recompute(self, dataset_with_errors):
        dataset_with_errors.metrics.wer()  # freeze original
        clone = dataset_with_errors.clone()
        clone.add("perfect", "perfect")
        # New example reflected in the recomputed metric.
        assert len(clone) == len(dataset_with_errors) + 1
        assert clone.metrics.wer().value is not None

    def test_clone_has_clean_metric_cache(self, sample_dataset):
        sample_dataset.metrics.wer()
        clone = sample_dataset.clone()
        assert clone.metrics._metric_cache == {}
