"""Tests for bewer.metrics._kt_stats module."""

import pytest

from bewer import Dataset
from bewer.alignment import Alignment, OpType
from bewer.metrics.base import METRIC_REGISTRY


class TestKTStatsExampleMetric:
    """Tests for _KTStats_ (ExampleMetric) class."""

    @pytest.fixture
    def dataset_correct(self):
        """Dataset where key term is correctly transcribed (TP=1, FN=0, FP=0)."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_error(self):
        """Dataset where key term is in ref but absent from hyp (TP=0, FN=1, FP=0)."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the dog jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    @pytest.fixture
    def dataset_fp(self):
        """Dataset where ref has key term once but hyp has it twice (TP=1, FN=0, FP=1)."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox fox jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    def test_num_ref_terms(self, dataset_correct):
        """Test num_ref_terms = 1 for a single key term in the reference."""
        stats = dataset_correct[0].metrics._kt_stats(vocab="animals")
        assert stats.num_ref_terms == 1

    def test_num_hyp_terms_correct(self, dataset_correct):
        """Test num_hyp_terms = 1 on an exact match."""
        stats = dataset_correct[0].metrics._kt_stats(vocab="animals")
        assert stats.num_hyp_terms == 1

    def test_num_hyp_terms_with_fp(self, dataset_fp):
        """Test num_hyp_terms = 2 when key term appears twice in hypothesis."""
        stats = dataset_fp[0].metrics._kt_stats(vocab="animals")
        assert stats.num_hyp_terms == 2

    def test_num_tp_correct(self, dataset_correct):
        """Test num_tp = 1 on correct transcription."""
        stats = dataset_correct[0].metrics._kt_stats(vocab="animals")
        assert stats.num_tp == 1

    def test_num_tp_error(self, dataset_error):
        """Test num_tp = 0 on incorrect transcription."""
        stats = dataset_error[0].metrics._kt_stats(vocab="animals")
        assert stats.num_tp == 0

    def test_num_fn_correct(self, dataset_correct):
        """Test num_fn = 0 on correct transcription."""
        stats = dataset_correct[0].metrics._kt_stats(vocab="animals")
        assert stats.num_fn == 0

    def test_num_fn_error(self, dataset_error):
        """Test num_fn = 1 on incorrect transcription."""
        stats = dataset_error[0].metrics._kt_stats(vocab="animals")
        assert stats.num_fn == 1

    def test_num_fp_none(self, dataset_correct):
        """Test num_fp = 0 on exact match."""
        stats = dataset_correct[0].metrics._kt_stats(vocab="animals")
        assert stats.num_fp == 0

    def test_num_fp_with_false_positive(self, dataset_fp):
        """Test num_fp = 1 when hyp has an extra key term occurrence."""
        stats = dataset_fp[0].metrics._kt_stats(vocab="animals")
        assert stats.num_fp == 1

    def test_tp_plus_fn_equals_ref_terms(self, dataset_correct):
        """Test invariant: TP + FN always equals num_ref_terms."""
        for fixture in [dataset_correct]:
            stats = fixture[0].metrics._kt_stats(vocab="animals")
            assert stats.num_tp + stats.num_fn == stats.num_ref_terms

    def test_tp_plus_fn_equals_ref_terms_on_error(self, dataset_error):
        """Test invariant holds when key term is missed."""
        stats = dataset_error[0].metrics._kt_stats(vocab="animals")
        assert stats.num_tp + stats.num_fn == stats.num_ref_terms


class TestKTStatsAlignmentAttributes:
    """Tests for tp_alignments, fn_alignments, fp_alignments on _KTStats_."""

    @pytest.fixture
    def dataset_correct(self):
        dataset = Dataset()
        dataset.add(ref="the fox jumps", hyp="the fox jumps", key_terms={"animals": ["fox"]})
        return dataset

    @pytest.fixture
    def dataset_error(self):
        dataset = Dataset()
        dataset.add(ref="the fox jumps", hyp="the dog jumps", key_terms={"animals": ["fox"]})
        return dataset

    @pytest.fixture
    def dataset_fp(self):
        dataset = Dataset()
        dataset.add(ref="the fox jumps", hyp="the fox fox jumps", key_terms={"animals": ["fox"]})
        return dataset

    def test_tp_alignments_correct(self, dataset_correct):
        stats = dataset_correct[0].metrics._kt_stats(vocab="animals")
        assert len(stats.tp_alignments) == 1
        assert all(op.type == OpType.MATCH for op in stats.tp_alignments[0])

    def test_fn_alignments_correct(self, dataset_correct):
        stats = dataset_correct[0].metrics._kt_stats(vocab="animals")
        assert stats.fn_alignments == []

    def test_fp_alignments_correct(self, dataset_correct):
        stats = dataset_correct[0].metrics._kt_stats(vocab="animals")
        assert stats.fp_alignments == []

    def test_tp_alignments_error(self, dataset_error):
        stats = dataset_error[0].metrics._kt_stats(vocab="animals")
        assert stats.tp_alignments == []

    def test_fn_alignments_error(self, dataset_error):
        stats = dataset_error[0].metrics._kt_stats(vocab="animals")
        assert len(stats.fn_alignments) == 1
        op_types = {op.type for op in stats.fn_alignments[0]}
        assert OpType.SUBSTITUTE in op_types

    def test_fp_alignments_fp_case(self, dataset_fp):
        stats = dataset_fp[0].metrics._kt_stats(vocab="animals")
        assert len(stats.tp_alignments) == 1
        assert len(stats.fp_alignments) == 1

    def test_count_invariants(self, dataset_correct, dataset_error, dataset_fp):
        for dataset in (dataset_correct, dataset_error, dataset_fp):
            stats = dataset[0].metrics._kt_stats(vocab="animals")
            assert len(stats.tp_alignments) == stats.num_tp
            assert len(stats.fn_alignments) == stats.num_fn
            assert len(stats.fp_alignments) == stats.num_fp

    def test_return_types(self, dataset_fp):
        stats = dataset_fp[0].metrics._kt_stats(vocab="animals")
        for seg in stats.tp_alignments + stats.fn_alignments + stats.fp_alignments:
            assert isinstance(seg, Alignment)

    def test_fp_subset_match_excluded(self):
        """Correctly transcribed subset hyp match is neither TP nor FP when allow_subsets=False."""
        dataset = Dataset()
        dataset.add(
            ref="hello world",
            hyp="hollow world",
            key_terms={"vocab": ["hello world", "world"]},
        )
        stats = dataset[0].metrics._kt_stats(vocab="vocab", allow_subsets=False)
        assert stats.fp_alignments == []
        assert stats.num_fp == 0
        assert stats.num_fn == 1

    def test_fp_simultaneous_fn_and_fp(self):
        """A key term substituted for another key term produces both an FN and an FP."""
        dataset = Dataset()
        dataset.add(
            ref="world",
            hyp="wall",
            key_terms={"vocab": ["world", "wall"]},
        )
        stats = dataset[0].metrics._kt_stats(vocab="vocab")
        assert len(stats.fn_alignments) == 1
        assert len(stats.fp_alignments) == 1
        assert stats.fn_alignments[0] == stats.fp_alignments[0]


class TestKTStatsDatasetMetric:
    """Tests for _KTStats (dataset-level Metric) class."""

    @pytest.fixture
    def mixed_dataset(self):
        """Dataset with one correct example and one missed key term."""
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox jumps",
            key_terms={"animals": ["fox"]},
        )
        dataset.add(
            ref="the rabbit runs",
            hyp="the dog runs",
            key_terms={"animals": ["rabbit"]},
        )
        return dataset

    def test_num_ref_terms_aggregates(self, mixed_dataset):
        """Test num_ref_terms sums example-level counts."""
        stats = mixed_dataset.metrics._kt_stats(vocab="animals")
        expected = sum(ex.metrics._kt_stats(vocab="animals").num_ref_terms for ex in mixed_dataset)
        assert stats.num_ref_terms == expected

    def test_num_hyp_terms_aggregates(self, mixed_dataset):
        """Test num_hyp_terms sums example-level counts."""
        stats = mixed_dataset.metrics._kt_stats(vocab="animals")
        expected = sum(ex.metrics._kt_stats(vocab="animals").num_hyp_terms for ex in mixed_dataset)
        assert stats.num_hyp_terms == expected

    def test_num_tp_aggregates(self, mixed_dataset):
        """Test num_tp sums example-level counts."""
        stats = mixed_dataset.metrics._kt_stats(vocab="animals")
        expected = sum(ex.metrics._kt_stats(vocab="animals").num_tp for ex in mixed_dataset)
        assert stats.num_tp == expected

    def test_num_fn_aggregates(self, mixed_dataset):
        """Test num_fn sums example-level counts."""
        stats = mixed_dataset.metrics._kt_stats(vocab="animals")
        expected = sum(ex.metrics._kt_stats(vocab="animals").num_fn for ex in mixed_dataset)
        assert stats.num_fn == expected

    def test_num_fp_aggregates(self, mixed_dataset):
        """Test num_fp sums example-level counts."""
        stats = mixed_dataset.metrics._kt_stats(vocab="animals")
        expected = sum(ex.metrics._kt_stats(vocab="animals").num_fp for ex in mixed_dataset)
        assert stats.num_fp == expected

    def test_empty_dataset(self):
        """Test _KTStats raises ValueError for unknown vocab."""
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset key term vocabularies"):
            dataset.metrics._kt_stats(vocab="animals").num_tp


class TestKTStatsIsPrivate:
    """Tests that _KTStats is hidden from the public metric listing."""

    def test_not_in_list_metrics(self):
        """Test _kt_stats is not visible in the default (non-private) metric listing."""
        public_names = [name for name in METRIC_REGISTRY.metric_classes if not name.startswith("_")]
        assert "_kt_stats" not in public_names

    def test_visible_with_show_private(self):
        """Test _kt_stats is present in the registry when private metrics are included."""
        assert "_kt_stats" in METRIC_REGISTRY.metric_classes


class TestKTStatsSharing:
    """Tests that all key term metrics share the same _KTStats instance for identical params."""

    @pytest.fixture
    def dataset(self):
        dataset = Dataset()
        dataset.add(
            ref="the fox jumps",
            hyp="the fox jumps",
            key_terms={"animals": ["fox"]},
        )
        return dataset

    def test_ktr_and_kter_share_instance(self, dataset):
        """KTR and KTER with identical params share the same _KTStats instance."""
        ktr = dataset.metrics.ktr(vocab="animals")
        kter = dataset.metrics.kter(vocab="animals")
        assert ktr._kt_stats is kter._kt_stats

    def test_ktr_and_ktp_share_instance(self, dataset):
        """KTR and KTP with identical params share the same _KTStats instance."""
        ktr = dataset.metrics.ktr(vocab="animals")
        ktp = dataset.metrics.ktp(vocab="animals")
        assert ktr._kt_stats is ktp._kt_stats

    def test_ktr_and_ktf_share_instance(self, dataset):
        """KTR and KTF with identical params share the same _KTStats instance."""
        ktr = dataset.metrics.ktr(vocab="animals")
        ktf = dataset.metrics.ktf(vocab="animals")
        assert ktr._kt_stats is ktf._kt_stats

    def test_all_four_share_instance(self, dataset):
        """KTER, KTR, KTP, and KTF all share the same _KTStats for identical params."""
        kter = dataset.metrics.kter(vocab="animals")
        ktr = dataset.metrics.ktr(vocab="animals")
        ktp = dataset.metrics.ktp(vocab="animals")
        ktf = dataset.metrics.ktf(vocab="animals")
        assert kter._kt_stats is ktr._kt_stats is ktp._kt_stats is ktf._kt_stats

    def test_different_vocabs_different_instances(self, dataset):
        """Different vocab params produce different _KTStats instances."""
        dataset.add_key_term_list("verbs", ["jumps"])
        ktr_animals = dataset.metrics.ktr(vocab="animals")
        ktr_verbs = dataset.metrics.ktr(vocab="verbs")
        assert ktr_animals._kt_stats is not ktr_verbs._kt_stats

    def test_different_normalized_different_instances(self, dataset):
        """Different normalized params produce different _KTStats instances."""
        ktr_norm = dataset.metrics.ktr(vocab="animals", normalized=True)
        ktr_unnorm = dataset.metrics.ktr(vocab="animals", normalized=False)
        assert ktr_norm._kt_stats is not ktr_unnorm._kt_stats

    def test_instance_is_cached(self, dataset):
        """Accessing _kt_stats twice on the same metric returns the identical object."""
        ktr = dataset.metrics.ktr(vocab="animals")
        assert ktr._kt_stats is ktr._kt_stats
