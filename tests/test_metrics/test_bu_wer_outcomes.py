"""Tests for the private _BUWEROutcomes metric shared by B-WER and U-WER."""

import warnings

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


class TestBUWEROutcomesMultiTokenModes:
    """Tests for the multi_token parameter modes."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    def test_split_counts_term_tokens_individually(self, dataset):
        """Split mode: every term token is an individual in-vocab word."""
        stats = dataset[3].metrics._bu_wer_outcomes(vocab="medical", multi_token="split")
        # "blood" alone is a term token, so the partial insert becomes biased.
        assert stats.num_biased_errors == 1
        assert stats.num_unbiased_errors == 0
        assert stats.num_ref_vocab_tokens == 0
        assert stats.num_ref_other_tokens == 1

    def test_flag_matches_occurrence_counts(self, dataset):
        """Flag mode gives the same counts as occurrence mode."""
        occurrence = dataset[3].metrics._bu_wer_outcomes(vocab="medical")
        flag = dataset[3].metrics._bu_wer_outcomes(vocab="medical", multi_token="flag")
        assert (
            flag.num_biased_errors,
            flag.num_unbiased_errors,
            flag.num_ref_vocab_tokens,
            flag.num_ref_other_tokens,
        ) == (
            occurrence.num_biased_errors,
            occurrence.num_unbiased_errors,
            occurrence.num_ref_vocab_tokens,
            occurrence.num_ref_other_tokens,
        )

    def test_flag_warns_about_multi_token_terms(self, dataset):
        """Flag mode warns, naming the multi-token vocabulary terms."""
        with pytest.warns(UserWarning, match="blood pressure"):
            dataset[3].metrics._bu_wer_outcomes(vocab="medical", multi_token="flag").num_biased_errors

    def test_occurrence_does_not_warn(self, dataset):
        """Occurrence mode stays silent about multi-token terms."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dataset[3].metrics._bu_wer_outcomes(vocab="medical").num_biased_errors

    def test_flag_silent_for_single_token_vocabulary(self):
        """Flag mode does not warn when every term tokenizes to a single token."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["fox"])))
        dataset.add(ref="the fox jumps", hyp="the dog jumps")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dataset[0].metrics._bu_wer_outcomes(vocab="animals", multi_token="flag").num_biased_errors

    def test_split_requires_no_term_occurrence(self, dataset):
        """Split-mode inserts: only word membership matters, not occurrence."""
        # Example 2 inserts the whole term: still biased in split mode.
        stats = dataset[2].metrics._bu_wer_outcomes(vocab="medical", multi_token="split")
        assert stats.num_biased_errors == 2
        assert stats.num_unbiased_errors == 0
        # Example 1 inserts two non-vocab words: still unbiased in split mode.
        stats = dataset[1].metrics._bu_wer_outcomes(vocab="medical", multi_token="split")
        assert stats.num_biased_errors == 0
        assert stats.num_unbiased_errors == 2


class TestBUWEROutcomesBackends:
    """Tests that both alignment backends produce identical attributions."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    @pytest.mark.parametrize("multi_token", ["occurrence", "split"])
    def test_error_align_matches_levenshtein(self, dataset, multi_token):
        """error_align attribution equals levenshtein attribution, occurrence and split."""
        lev = dataset.metrics._bu_wer_outcomes(vocab="medical", alignment="levenshtein", multi_token=multi_token)
        ea = dataset.metrics._bu_wer_outcomes(vocab="medical", alignment="error_align", multi_token=multi_token)
        assert (
            ea.num_biased_errors,
            ea.num_unbiased_errors,
            ea.num_ref_vocab_tokens,
            ea.num_ref_other_tokens,
        ) == (
            lev.num_biased_errors,
            lev.num_unbiased_errors,
            lev.num_ref_vocab_tokens,
            lev.num_ref_other_tokens,
        )

    def test_error_align_per_example(self, dataset):
        """Per-example error_align attribution matches levenshtein (multi-word inserts)."""
        for i in range(len(dataset)):
            lev = dataset[i].metrics._bu_wer_outcomes(vocab="medical", alignment="levenshtein")
            ea = dataset[i].metrics._bu_wer_outcomes(vocab="medical", alignment="error_align")
            assert (
                ea.num_biased_errors,
                ea.num_unbiased_errors,
                ea.num_ref_vocab_tokens,
                ea.num_ref_other_tokens,
            ) == (
                lev.num_biased_errors,
                lev.num_unbiased_errors,
                lev.num_ref_vocab_tokens,
                lev.num_ref_other_tokens,
            )

    def test_unnormalized_error_align(self):
        """normalized=False works with the error_align backend."""
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("animals", terms=["Fox"])))
        dataset.add(ref="The Fox jumps", hyp="The fox jumps")
        for alignment in ("levenshtein", "error_align"):
            stats = dataset[0].metrics._bu_wer_outcomes(vocab="animals", normalized=False, alignment=alignment)
            # "Fox" (vocab) was substituted by "fox": biased error, biased denominator.
            assert stats.num_biased_errors == 1
            assert stats.num_unbiased_errors == 0
            assert stats.num_ref_vocab_tokens == 1
            assert stats.num_ref_other_tokens == 2


class TestBUWEROutcomesDecomposition:
    """Tests that B/U statistics always form an exact split of plain WER."""

    @pytest.fixture
    def dataset(self):
        return _medical_dataset()

    @pytest.mark.parametrize("multi_token", ["occurrence", "split", "flag"])
    @pytest.mark.parametrize("alignment", ["levenshtein", "error_align"])
    def test_exact_split_of_wer(self, dataset, multi_token, alignment):
        """Biased + unbiased errors equal WER edits; denominators equal reference length."""
        stats = dataset.metrics._bu_wer_outcomes(vocab="medical", multi_token=multi_token, alignment=alignment)
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

    def test_invalid_alignment(self, dataset):
        """Alignment backend must be 'levenshtein' or 'error_align'."""
        with pytest.raises(ValueError, match="alignment must be"):
            dataset.metrics._bu_wer_outcomes(vocab="medical", alignment="bogus")

    def test_invalid_multi_token(self, dataset):
        """Multi-token mode must be 'occurrence', 'flag', or 'split'."""
        with pytest.raises(ValueError, match="multi_token must be"):
            dataset.metrics._bu_wer_outcomes(vocab="medical", multi_token="bogus")
