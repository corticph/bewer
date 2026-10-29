"""Tests for the private _BUWEROutcomes metric shared by B-WER and U-WER."""

import pytest

from bewer import Dataset, Vocabulary, get_config
from bewer.metrics.base import METRIC_REGISTRY


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


class TestBUWEROutcomesAttribution:
    """Tests for _BUWEROutcomes_ (ExampleMetric) attribution under occurrence semantics."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_substitution_of_vocab_token(self, dataset):
        """Substituted vocab token: biased error plus biased denominator."""
        stats = dataset[0].metrics._bu_wer_outcomes(vocab="medical")
        assert stats.num_biased_errors == 1
        assert stats.num_unbiased_errors == 0
        assert stats.num_ref_vocab_tokens == 1
        assert stats.num_ref_other_tokens == 2

    def test_non_vocab_insertions(self, dataset):
        """Inserted non-vocab words: unbiased errors, no reference-side counts."""
        stats = dataset[1].metrics._bu_wer_outcomes(vocab="medical")
        assert stats.num_biased_errors == 0
        assert stats.num_unbiased_errors == 2
        assert stats.num_ref_vocab_tokens == 0
        assert stats.num_ref_other_tokens == 2

    def test_inserted_term_occurrence(self, dataset):
        """A fully inserted multi-token term: every insert is a biased error."""
        stats = dataset[2].metrics._bu_wer_outcomes(vocab="medical")
        assert stats.num_biased_errors == 2
        assert stats.num_unbiased_errors == 0
        assert stats.num_ref_vocab_tokens == 0
        assert stats.num_ref_other_tokens == 1

    def test_partial_term_insert_is_not_biased(self, dataset):
        """A partial term insert is unbiased: tokens count only within a full match."""
        stats = dataset[3].metrics._bu_wer_outcomes(vocab="medical")
        assert stats.num_biased_errors == 0
        assert stats.num_unbiased_errors == 1
        assert stats.num_ref_vocab_tokens == 0
        assert stats.num_ref_other_tokens == 1

    def test_deletion_of_vocab_token(self):
        """Deleted vocab token: biased error plus biased denominator."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["fox"])))
        dataset.add(ref="the fox jumps", hyp="the jumps")
        stats = dataset[0].metrics._bu_wer_outcomes(vocab="animals")
        assert stats.num_biased_errors == 1
        assert stats.num_unbiased_errors == 0
        assert stats.num_ref_vocab_tokens == 1
        assert stats.num_ref_other_tokens == 2

    def test_substitution_of_non_vocab_token(self):
        """Substituted non-vocab token: unbiased error plus unbiased denominator."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["fox"])))
        dataset.add(ref="the dog house", hyp="the cat house")
        stats = dataset[0].metrics._bu_wer_outcomes(vocab="animals")
        assert stats.num_biased_errors == 0
        assert stats.num_unbiased_errors == 1
        assert stats.num_ref_vocab_tokens == 0
        assert stats.num_ref_other_tokens == 3

    def test_perfect_match_attribution(self):
        """Perfect match of a multi-token term: biased denominator, no errors."""
        dataset = Dataset(
            config=get_config("base").clone().add_vocabulary(Vocabulary("medical", terms=["blood pressure"]))
        )
        dataset.add(ref="blood pressure is normal", hyp="blood pressure is normal")
        stats = dataset[0].metrics._bu_wer_outcomes(vocab="medical")
        assert stats.num_biased_errors == 0
        assert stats.num_unbiased_errors == 0
        assert stats.num_ref_vocab_tokens == 2
        assert stats.num_ref_other_tokens == 2

    def test_empty_reference(self):
        """Empty reference: all inserts land on the unbiased side, zero denominators."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["fox"])))
        dataset.add(ref="", hyp="hello world")
        stats = dataset[0].metrics._bu_wer_outcomes(vocab="animals")
        assert stats.num_biased_errors == 0
        assert stats.num_unbiased_errors == 2
        assert stats.num_ref_vocab_tokens == 0
        assert stats.num_ref_other_tokens == 0


class TestBUWEROutcomesDecomposition:
    """Tests that B/U statistics always form an exact split of plain WER."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_exact_split_of_wer(self, dataset):
        """Biased + unbiased errors equal WER edits; denominators equal reference length."""
        stats = dataset.metrics._bu_wer_outcomes(vocab="medical")
        wer = dataset.metrics.wer()
        assert stats.num_biased_errors + stats.num_unbiased_errors == wer.num_edits
        assert stats.num_ref_vocab_tokens + stats.num_ref_other_tokens == wer.ref_length


class TestBUWEROutcomesDatasetMetric:
    """Tests for the dataset-level _BUWEROutcomes aggregation."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_aggregates_examples(self, dataset):
        """Dataset totals are the sum of example-level statistics."""
        stats = dataset.metrics._bu_wer_outcomes(vocab="medical")
        expected = [
            sum(ex.metrics._bu_wer_outcomes(vocab="medical").num_biased_errors for ex in dataset),
            sum(ex.metrics._bu_wer_outcomes(vocab="medical").num_unbiased_errors for ex in dataset),
            sum(ex.metrics._bu_wer_outcomes(vocab="medical").num_ref_vocab_tokens for ex in dataset),
            sum(ex.metrics._bu_wer_outcomes(vocab="medical").num_ref_other_tokens for ex in dataset),
        ]
        assert (
            [stats.num_biased_errors, stats.num_unbiased_errors, stats.num_ref_vocab_tokens, stats.num_ref_other_tokens]
            == expected
            == [3, 3, 1, 6]
        )

    def test_empty_dataset(self):
        """Statistics on an empty dataset are zero."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["fox"])))
        dataset.metrics._bu_wer_outcomes(vocab="animals")
        stats = dataset.metrics._bu_wer_outcomes(vocab="animals")
        assert stats.num_biased_errors == 0
        assert stats.num_unbiased_errors == 0
        assert stats.num_ref_vocab_tokens == 0
        assert stats.num_ref_other_tokens == 0


class TestBUWEROutcomesIsPrivate:
    """Tests that _BUWEROutcomes is hidden from the public metric listing."""

    def test_not_in_list_metrics(self):
        """_bu_wer_outcomes is not visible in the default (non-private) metric listing."""
        public_names = [name for name in METRIC_REGISTRY.metric_classes if not name.startswith("_")]
        assert "_bu_wer_outcomes" not in public_names

    def test_visible_with_show_private(self):
        """_bu_wer_outcomes is present in the registry when private metrics are included."""
        assert "_bu_wer_outcomes" in METRIC_REGISTRY.metric_classes


class TestBUWEROutcomesSharing:
    """Tests that B-WER and U-WER share the same _BUWEROutcomes instance for identical params."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_b_wer_and_u_wer_share_instance(self, dataset):
        """BWER and UWER with identical params share the same _BUWEROutcomes instance."""
        b_wer = dataset.metrics.b_wer(vocab="medical")
        u_wer = dataset.metrics.u_wer(vocab="medical")
        assert b_wer._bu_wer_outcomes is u_wer._bu_wer_outcomes

    def test_different_vocabs_different_instances(self):
        """Different vocab params produce different _BUWEROutcomes instances."""
        config = get_config("base").clone()
        config = config.add_vocabulary(Vocabulary("animals", terms=["fox"]))
        config = config.add_vocabulary(Vocabulary("verbs", terms=["jumps"]))
        dataset = Dataset(config=config)
        dataset.add(ref="the fox jumps", hyp="the fox jumps")
        assert (
            dataset.metrics.b_wer(vocab="animals")._bu_wer_outcomes
            is not dataset.metrics.b_wer(vocab="verbs")._bu_wer_outcomes
        )

    def test_different_normalized_different_instances(self, dataset):
        """Different normalized params produce different _BUWEROutcomes instances."""
        norm = dataset.metrics.b_wer(vocab="medical", normalized=True)
        unnorm = dataset.metrics.b_wer(vocab="medical", normalized=False)
        assert norm._bu_wer_outcomes is not unnorm._bu_wer_outcomes

    def test_instance_is_cached(self, dataset):
        """Accessing _bu_wer_outcomes twice on the same metric returns the identical object."""
        b_wer = dataset.metrics.b_wer(vocab="medical")
        assert b_wer._bu_wer_outcomes is b_wer._bu_wer_outcomes


class TestBUWEROutcomesParameters:
    """Tests for _BUWEROutcomes parameter validation."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_missing_required_vocab(self, dataset):
        """Metric construction fails without the required vocab parameter."""
        with pytest.raises(ValueError, match="vocab"):
            dataset.metrics._bu_wer_outcomes()

    def test_unknown_vocab(self, dataset):
        """Metric construction fails for a vocabulary not attached to the dataset."""
        with pytest.raises(ValueError, match="Vocabulary 'animals' not found"):
            dataset.metrics._bu_wer_outcomes(vocab="animals")
