"""Tests for the U-WER (u_wer) metric."""

import pytest

from bewer import Dataset, Vocabulary, get_config
from bewer.metrics.key_term.u_wer import UWER, UWER_


def _medical_dataset():
    """A mixed dataset over a single-token and a multi-token vocabulary term."""
    dataset = Dataset(
        config=get_config("base").clone().add_vocabulary(Vocabulary("medical", terms=["blood pressure", "fox"]))
    )
    dataset.add(ref="the fox jumps", hyp="the dog jumps")  # substitution of a vocab token
    dataset.add(ref="hello there", hyp="the man hello there")  # two non-vocab insertions
    dataset.add(ref="hello", hyp="blood pressure hello")  # full multi-token term inserted
    dataset.add(ref="hello", hyp="blood hello")  # partial term insert
    return dataset


class TestUWERExampleMetric:
    """Tests for UWER_ (ExampleMetric) values."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_substitution_of_vocab_token(self, dataset):
        """A vocab-token substitution leaves U-WER at zero errors."""
        u_wer = dataset[0].metrics.u_wer(vocab="medical")
        assert u_wer.num_errors == 0
        assert u_wer.ref_length == 2
        assert u_wer.value == 0.0

    def test_non_vocab_insertions(self, dataset):
        """Non-vocab insertions count as unbiased errors."""
        u_wer = dataset[1].metrics.u_wer(vocab="medical")
        assert u_wer.num_errors == 2
        assert u_wer.ref_length == 2
        assert u_wer.value == 1.0

    def test_inserted_term_occurrence(self, dataset):
        """A fully inserted term does not touch the unbiased side."""
        u_wer = dataset[2].metrics.u_wer(vocab="medical")
        assert u_wer.num_errors == 0
        assert u_wer.ref_length == 1
        assert u_wer.value == 0.0

    def test_partial_term_insert_unbiased(self, dataset):
        """Occurrence semantics: a partial term insert is an unbiased error."""
        u_wer = dataset[3].metrics.u_wer(vocab="medical")
        assert u_wer.num_errors == 1
        assert u_wer.ref_length == 1
        assert u_wer.value == 1.0

    def test_partial_term_insert_excluded_in_split_mode(self, dataset):
        """Split semantics: a partial term insert no longer an unbiased error."""
        u_wer = dataset[3].metrics.u_wer(vocab="medical", multi_token="split")
        assert u_wer.num_errors == 0
        assert u_wer.ref_length == 1
        assert u_wer.value == 0.0

    def test_substitution_of_non_vocab_token(self):
        """A non-vocab substitution counts as an unbiased error."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["fox"])))
        dataset.add(ref="the dog house", hyp="the cat house")
        u_wer = dataset[0].metrics.u_wer(vocab="animals")
        assert u_wer.num_errors == 1
        assert u_wer.ref_length == 3
        assert u_wer.value == 1 / 3


class TestUWERZeroDenominators:
    """Tests for the empty-denominator convention (mirrors plain WER)."""

    def test_empty_ref_returns_num_errors(self):
        """With no non-vocab reference tokens, the value is the raw unbiased-error count."""
        dataset = Dataset(
            config=get_config("base").clone().add_vocabulary(Vocabulary("medical", terms=["blood pressure"]))
        )
        dataset.add(ref="", hyp="hello world")
        u_wer = dataset[0].metrics.u_wer(vocab="medical")
        assert u_wer.ref_length == 0
        assert u_wer.value == float(u_wer.num_errors) == 2.0


class TestUWERDatasetMetric:
    """Tests for UWER (dataset-level Metric) aggregation."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_num_errors_aggregates(self, dataset):
        """Dataset num_errors sums the example-level values."""
        u_wer = dataset.metrics.u_wer(vocab="medical")
        assert u_wer.num_errors == 3
        expected = sum(ex.metrics.u_wer(vocab="medical").num_errors for ex in dataset)
        assert u_wer.num_errors == expected

    def test_ref_length_aggregates(self, dataset):
        """Dataset ref_length sums the example-level values."""
        u_wer = dataset.metrics.u_wer(vocab="medical")
        assert u_wer.ref_length == 6
        expected = sum(ex.metrics.u_wer(vocab="medical").ref_length for ex in dataset)
        assert u_wer.ref_length == expected

    def test_value_calculation(self, dataset):
        """Value is num_errors over ref_length."""
        u_wer = dataset.metrics.u_wer(vocab="medical")
        assert u_wer.value == u_wer.num_errors / u_wer.ref_length == 0.5

    def test_empty_dataset(self):
        """An empty dataset yields zero errors and zero length."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["fox"])))
        u_wer = dataset.metrics.u_wer(vocab="animals")
        assert u_wer.ref_length == 0
        assert u_wer.num_errors == 0


class TestUWERSplitOfWER:
    """Tests that U-WER pairs with B-WER into an exact split of plain WER."""

    @pytest.mark.parametrize("multi_token", ["occurrence", "split"])
    @pytest.mark.parametrize("alignment", ["levenshtein", "error_align"])
    def test_errors_split_wer(self, multi_token, alignment):
        """U-WER + B-WER errors equal plain WER edits, on both backends and modes."""
        dataset = _medical_dataset()
        u_wer = dataset.metrics.u_wer(vocab="medical", multi_token=multi_token, alignment=alignment)
        b_wer = dataset.metrics.b_wer(vocab="medical", multi_token=multi_token, alignment=alignment)
        wer = dataset.metrics.wer()
        assert u_wer.num_errors + b_wer.num_errors == wer.num_edits
        assert u_wer.ref_length + b_wer.ref_length == wer.ref_length


class TestUWERMetricAttributes:
    """Tests for UWER metric attributes."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_short_name(self, dataset):
        """UWER reports its registered display name."""
        u_wer = dataset.metrics.u_wer(vocab="medical")
        assert u_wer.short_name_base == "U-WER"

    def test_long_name(self, dataset):
        """UWER reports its long display name."""
        u_wer = dataset.metrics.u_wer(vocab="medical")
        assert u_wer.long_name_base == "Unbiased Word Error Rate"

    def test_description(self, dataset):
        """UWER has a non-empty description."""
        u_wer = dataset.metrics.u_wer(vocab="medical")
        assert len(u_wer.description) > 0

    def test_example_cls(self, dataset):
        """UWER's example_cls is UWER_."""
        u_wer = dataset.metrics.u_wer(vocab="medical")
        assert u_wer.example_cls == UWER_


class TestUWERMetricValues:
    """Tests for UWER.metric_values() introspection."""

    def test_metric_values_main(self):
        """Value is the main metric value."""
        values = UWER.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        """num_errors and ref_length are other metric values."""
        values = UWER.metric_values()
        assert "num_errors" in values["other"]
        assert "ref_length" in values["other"]

    def test_example_metric_values(self):
        """UWER_ value: main value; num_errors/ref_length are other values."""
        values = UWER_.metric_values()
        assert values["main"] == "value"
        assert "num_errors" in values["other"]
        assert "ref_length" in values["other"]


class TestUWERParameters:
    """Tests for UWER parameter validation."""

    def test_missing_required_vocab(self):
        """Construction fails without the required vocab parameter."""
        dataset = _medical_dataset()
        with pytest.raises(ValueError, match="vocab"):
            dataset.metrics.u_wer()

    def test_unknown_vocab(self):
        """Construction fails for a vocabulary not attached to the dataset."""
        dataset = _medical_dataset()
        with pytest.raises(ValueError, match="Vocabulary 'animals' not found"):
            dataset.metrics.u_wer(vocab="animals")

    def test_invalid_alignment(self):
        """Construction fails for an unknown alignment backend."""
        dataset = _medical_dataset()
        with pytest.raises(ValueError, match="alignment must be"):
            dataset.metrics.u_wer(vocab="medical", alignment="bogus")


class TestUWERFlagWarning:
    """Tests that flag-mode warnings surface through UWER."""

    def test_flag_warns_for_multi_token_vocabulary(self):
        """Flag mode warns, naming the multi-token terms."""
        dataset = _medical_dataset()
        with pytest.warns(UserWarning, match="blood pressure"):
            dataset.metrics.u_wer(vocab="medical", multi_token="flag").value
