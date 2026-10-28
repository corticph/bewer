"""Tests for the B-WER (b_wer) metric."""

import pytest

from bewer import Dataset, Vocabulary, get_config
from bewer.metrics.general.b_wer import BWER, BWER_


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


class TestBWERExampleMetric:
    """Tests for BWER_ (ExampleMetric) values."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_substitution_of_vocab_token(self, dataset):
        """B-WER counts a substituted vocab token: 1 error over 1 vocab ref token."""
        b_wer = dataset[0].metrics.b_wer(vocab="medical")
        assert b_wer.num_errors == 1
        assert b_wer.ref_length == 1
        assert b_wer.value == 1.0

    def test_non_vocab_insertions(self, dataset):
        """Non-vocab insertions leave B-WER at zero, with an empty denominator."""
        b_wer = dataset[1].metrics.b_wer(vocab="medical")
        assert b_wer.num_errors == 0
        assert b_wer.ref_length == 0

    def test_inserted_term_occurrence(self, dataset):
        """A fully inserted term gives zero-denominator B-WER: value equals the error count."""
        b_wer = dataset[2].metrics.b_wer(vocab="medical")
        assert b_wer.num_errors == 2
        assert b_wer.ref_length == 0
        assert b_wer.value == 2.0

    def test_partial_term_insert_unbiased(self, dataset):
        """Occurrence semantics: a partial term insert is not a biased error."""
        b_wer = dataset[3].metrics.b_wer(vocab="medical")
        assert b_wer.num_errors == 0

    def test_partial_term_insert_biased_in_split_mode(self, dataset):
        """Split semantics: a partial term insert becomes a biased error."""
        b_wer = dataset[3].metrics.b_wer(vocab="medical", multi_token="split")
        assert b_wer.num_errors == 1
        assert b_wer.ref_length == 0
        assert b_wer.value == 1.0

    def test_deletion_of_vocab_token(self):
        """B-WER counts a deleted vocab token."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["fox"])))
        dataset.add(ref="the fox jumps", hyp="the jumps")
        b_wer = dataset[0].metrics.b_wer(vocab="animals")
        assert b_wer.num_errors == 1
        assert b_wer.ref_length == 1
        assert b_wer.value == 1.0

    def test_perfect_match(self):
        """No errors on a perfect match of a multi-token term."""
        dataset = Dataset(
            config=get_config("base").clone().add_vocabulary(Vocabulary("medical", terms=["blood pressure"]))
        )
        dataset.add(ref="blood pressure is normal", hyp="blood pressure is normal")
        b_wer = dataset[0].metrics.b_wer(vocab="medical")
        assert b_wer.num_errors == 0
        assert b_wer.ref_length == 2
        assert b_wer.value == 0.0


class TestBWERZeroDenominators:
    """Tests for the empty-denominator convention (mirrors plain WER)."""

    def test_empty_ref_returns_num_errors(self):
        """With no biased reference tokens, the value is the raw biased-error count."""
        dataset = Dataset(
            config=get_config("base").clone().add_vocabulary(Vocabulary("medical", terms=["blood pressure"]))
        )
        dataset.add(ref="", hyp="blood pressure")
        b_wer = dataset[0].metrics.b_wer(vocab="medical")
        assert b_wer.ref_length == 0
        assert b_wer.value == float(b_wer.num_errors) == 2.0


class TestBWERDatasetMetric:
    """Tests for BWER (dataset-level Metric) aggregation."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_num_errors_aggregates(self, dataset):
        """Dataset num_errors sums the example-level values."""
        b_wer = dataset.metrics.b_wer(vocab="medical")
        assert b_wer.num_errors == 3
        expected = sum(ex.metrics.b_wer(vocab="medical").num_errors for ex in dataset)
        assert b_wer.num_errors == expected

    def test_ref_length_aggregates(self, dataset):
        """Dataset ref_length sums the example-level values."""
        b_wer = dataset.metrics.b_wer(vocab="medical")
        assert b_wer.ref_length == 1
        expected = sum(ex.metrics.b_wer(vocab="medical").ref_length for ex in dataset)
        assert b_wer.ref_length == expected

    def test_value_calculation(self, dataset):
        """Value is num_errors over ref_length."""
        b_wer = dataset.metrics.b_wer(vocab="medical")
        assert b_wer.value == b_wer.num_errors / b_wer.ref_length == 3.0

    def test_empty_dataset(self):
        """An empty dataset yields zero errors and zero length."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["fox"])))
        dataset.metrics.b_wer(vocab="animals")
        b_wer = dataset.metrics.b_wer(vocab="animals")
        assert b_wer.ref_length == 0
        assert b_wer.num_errors == 0


class TestBWERSplitOfWER:
    """Tests that B-WER and U-WER together form an exact split of plain WER."""

    @pytest.mark.parametrize("multi_token", ["occurrence", "split"])
    @pytest.mark.parametrize("alignment", ["levenshtein", "error_align"])
    def test_errors_split_wer(self, multi_token, alignment):
        """B-WER + U-WER errors equal plain WER edits, on both backends and modes."""
        dataset = _medical_dataset()
        b_wer = dataset.metrics.b_wer(vocab="medical", multi_token=multi_token, alignment=alignment)
        u_wer = dataset.metrics.u_wer(vocab="medical", multi_token=multi_token, alignment=alignment)
        wer = dataset.metrics.wer()
        assert b_wer.num_errors + u_wer.num_errors == wer.num_edits
        assert b_wer.ref_length + u_wer.ref_length == wer.ref_length


class TestBWERMetricAttributes:
    """Tests for BWER metric attributes."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_short_name(self, dataset):
        """BWER reports its registered display name."""
        b_wer = dataset.metrics.b_wer(vocab="medical")
        assert b_wer.short_name_base == "B-WER"

    def test_long_name(self, dataset):
        """BWER reports its long display name."""
        b_wer = dataset.metrics.b_wer(vocab="medical")
        assert b_wer.long_name_base == "Biased Word Error Rate"

    def test_description(self, dataset):
        """BWER has a non-empty description."""
        b_wer = dataset.metrics.b_wer(vocab="medical")
        assert len(b_wer.description) > 0

    def test_example_cls(self, dataset):
        """BWER's example_cls is BWER_."""
        b_wer = dataset.metrics.b_wer(vocab="medical")
        assert b_wer.example_cls == BWER_


class TestBWERMetricValues:
    """Tests for BWER.metric_values() introspection."""

    def test_metric_values_main(self):
        """Value is the main metric value."""
        values = BWER.metric_values()
        assert values["main"] == "value"

    def test_metric_values_other(self):
        """num_errors and ref_length are other metric values."""
        values = BWER.metric_values()
        assert "num_errors" in values["other"]
        assert "ref_length" in values["other"]

    def test_example_metric_values(self):
        """BWER_ value: main value; num_errors/ref_length are other values."""
        values = BWER_.metric_values()
        assert values["main"] == "value"
        assert "num_errors" in values["other"]
        assert "ref_length" in values["other"]


class TestBWERParameters:
    """Tests for BWER parameter validation."""

    def test_missing_required_vocab(self):
        """Construction fails without the required vocab parameter."""
        dataset = _medical_dataset()
        with pytest.raises(ValueError, match="vocab"):
            dataset.metrics.b_wer()

    def test_unknown_vocab(self):
        """Construction fails for a vocabulary not attached to the dataset."""
        dataset = _medical_dataset()
        with pytest.raises(ValueError, match="Vocabulary 'animals' not found"):
            dataset.metrics.b_wer(vocab="animals")

    def test_invalid_multi_token(self):
        """Construction fails for an unknown multi_token mode."""
        dataset = _medical_dataset()
        with pytest.raises(ValueError, match="multi_token must be"):
            dataset.metrics.b_wer(vocab="medical", multi_token="bogus")


class TestBWERFlagWarning:
    """Tests that flag-mode warnings surface through BWER."""

    def test_flag_warns_for_multi_token_vocabulary(self):
        """Flag mode warns, naming the multi-token terms."""
        dataset = _medical_dataset()
        with pytest.warns(UserWarning, match="blood pressure"):
            dataset.metrics.b_wer(vocab="medical", multi_token="flag").value
