"""Tests for bewer.metrics._alphanum_stats module."""

import pytest

from bewer import Dataset
from bewer.alignment import Alignment, OpType
from bewer.metrics.base import METRIC_REGISTRY


class TestAlphaNumStatsExampleMetric:
    """Tests for _AlphaNumStats_ (ExampleMetric)."""

    @pytest.fixture
    def dataset_correct(self):
        """Abbreviation correctly transcribed (TP=1, FN=0, FP=0)."""
        dataset = Dataset()
        dataset.add(ref="patient had an MRI yesterday", hyp="patient had an MRI yesterday")
        return dataset

    @pytest.fixture
    def dataset_missed(self):
        """Abbreviation in ref but absent from hyp (TP=0, FN=1, FP=0)."""
        dataset = Dataset()
        dataset.add(ref="patient had an MRI yesterday", hyp="patient had an exam yesterday")
        return dataset

    @pytest.fixture
    def dataset_lowercased(self):
        """Abbreviation lowercased in hyp — case is the signal so this is a miss (TP=0, FN=1, FP=1).
        Both ref MRI and hyp mri match the regex (the second branch catches lowercase letter+digit-free
        tokens? No — `mri` has no digit and no uppercase, so it does NOT match. So FP=0)."""
        dataset = Dataset()
        dataset.add(ref="patient had an MRI yesterday", hyp="patient had an mri yesterday")
        return dataset

    @pytest.fixture
    def dataset_spurious(self):
        """Abbreviation only in hyp (TP=0, FN=0, FP=1)."""
        dataset = Dataset()
        dataset.add(ref="patient had an exam yesterday", hyp="patient had an MRI yesterday")
        return dataset

    @pytest.fixture
    def dataset_mixed(self):
        """Two abbreviations in ref, one correct (TP=1, FN=1, FP=0)."""
        dataset = Dataset()
        dataset.add(ref="MRI shows HbA1c elevated", hyp="MRI shows haemoglobin elevated")
        return dataset

    def test_num_ref_terms_correct(self, dataset_correct):
        stats = dataset_correct[0].metrics._alphanum_stats()
        assert stats.num_ref_terms == 1

    def test_num_ref_terms_mixed(self, dataset_mixed):
        stats = dataset_mixed[0].metrics._alphanum_stats()
        assert stats.num_ref_terms == 2

    def test_num_tp_correct(self, dataset_correct):
        stats = dataset_correct[0].metrics._alphanum_stats()
        assert stats.num_tp == 1
        assert stats.num_fn == 0
        assert stats.num_fp == 0

    def test_missed_abbreviation_is_fn(self, dataset_missed):
        stats = dataset_missed[0].metrics._alphanum_stats()
        assert stats.num_tp == 0
        assert stats.num_fn == 1
        assert stats.num_fp == 0

    def test_lowercased_abbreviation_is_fn(self, dataset_lowercased):
        """ref MRI vs hyp mri: ref matches regex, hyp does NOT (no case signal, no digit) → FN only."""
        stats = dataset_lowercased[0].metrics._alphanum_stats()
        assert stats.num_tp == 0
        assert stats.num_fn == 1
        assert stats.num_fp == 0

    def test_spurious_abbreviation_is_fp(self, dataset_spurious):
        stats = dataset_spurious[0].metrics._alphanum_stats()
        assert stats.num_tp == 0
        assert stats.num_fn == 0
        assert stats.num_fp == 1

    def test_mixed_counts(self, dataset_mixed):
        stats = dataset_mixed[0].metrics._alphanum_stats()
        assert stats.num_tp == 1
        assert stats.num_fn == 1
        assert stats.num_fp == 0

    @pytest.mark.parametrize(
        "fixture_name",
        ["dataset_correct", "dataset_missed", "dataset_lowercased", "dataset_spurious", "dataset_mixed"],
    )
    def test_tp_plus_fn_equals_ref_terms(self, request, fixture_name):
        """Invariant: TP + FN == num_ref_terms for every example."""
        dataset = request.getfixturevalue(fixture_name)
        stats = dataset[0].metrics._alphanum_stats()
        assert stats.num_tp + stats.num_fn == stats.num_ref_terms


class TestAlphaNumStatsAlignmentAttributes:
    """Tests for tp/fn/fp alignment lists on _AlphaNumStats_."""

    def test_tp_alignments_all_match(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        stats = dataset[0].metrics._alphanum_stats()
        assert len(stats.tp_alignments) == 1
        assert all(op.type == OpType.MATCH for op in stats.tp_alignments[0])

    def test_fn_alignment_has_edit(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an exam")
        stats = dataset[0].metrics._alphanum_stats()
        assert len(stats.fn_alignments) == 1
        op_types = {op.type for op in stats.fn_alignments[0]}
        assert OpType.SUBSTITUTE in op_types

    def test_fp_alignment_has_edit(self):
        dataset = Dataset()
        dataset.add(ref="patient had an exam", hyp="patient had an MRI")
        stats = dataset[0].metrics._alphanum_stats()
        assert len(stats.fp_alignments) == 1

    def test_invariant_alignment_counts(self):
        dataset = Dataset()
        dataset.add(ref="MRI shows HbA1c elevated", hyp="MRI shows haemoglobin elevated")
        stats = dataset[0].metrics._alphanum_stats()
        assert len(stats.tp_alignments) == stats.num_tp
        assert len(stats.fn_alignments) == stats.num_fn
        assert len(stats.fp_alignments) == stats.num_fp

    def test_alignment_segments_are_Alignment(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had no exam")
        stats = dataset[0].metrics._alphanum_stats()
        for seg in stats.tp_alignments + stats.fn_alignments + stats.fp_alignments:
            assert isinstance(seg, Alignment)


class TestAlphaNumStatsDatasetMetric:
    """Tests for the dataset-level _AlphaNumStats aggregation."""

    @pytest.fixture
    def mixed_dataset(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI yesterday", hyp="patient had an MRI yesterday")
        dataset.add(ref="HbA1c was elevated", hyp="haemoglobin was elevated")
        return dataset

    def test_num_ref_terms_aggregates(self, mixed_dataset):
        stats = mixed_dataset.metrics._alphanum_stats()
        expected = sum(ex.metrics._alphanum_stats().num_ref_terms for ex in mixed_dataset)
        assert stats.num_ref_terms == expected

    def test_num_tp_aggregates(self, mixed_dataset):
        stats = mixed_dataset.metrics._alphanum_stats()
        expected = sum(ex.metrics._alphanum_stats().num_tp for ex in mixed_dataset)
        assert stats.num_tp == expected

    def test_num_fn_aggregates(self, mixed_dataset):
        stats = mixed_dataset.metrics._alphanum_stats()
        expected = sum(ex.metrics._alphanum_stats().num_fn for ex in mixed_dataset)
        assert stats.num_fn == expected

    def test_num_fp_aggregates(self, mixed_dataset):
        stats = mixed_dataset.metrics._alphanum_stats()
        expected = sum(ex.metrics._alphanum_stats().num_fp for ex in mixed_dataset)
        assert stats.num_fp == expected


class TestAlphaNumStatsCustomPattern:
    """Tests for user-supplied regex patterns."""

    def test_custom_pattern_matches_subset(self):
        dataset = Dataset()
        dataset.add(ref="MRI and ECG", hyp="MRI and ECG")
        # Only match "MRI" literally
        stats = dataset[0].metrics._alphanum_stats(pattern=r"MRI")
        assert stats.num_ref_terms == 1

    def test_invalid_pattern_raises(self):
        dataset = Dataset()
        dataset.add(ref="MRI", hyp="MRI")
        with pytest.raises(ValueError, match="Invalid alphanumerical regex pattern"):
            dataset.metrics._alphanum_stats(pattern=r"[unclosed").num_ref_terms


class TestAlphaNumStatsIsPrivate:
    """Tests that _AlphaNumStats is hidden from the public metric listing."""

    def test_not_in_public_metric_listing(self):
        public_names = [name for name in METRIC_REGISTRY.metric_classes if not name.startswith("_")]
        assert "_alphanum_stats" not in public_names

    def test_visible_in_full_registry(self):
        assert "_alphanum_stats" in METRIC_REGISTRY.metric_classes


class TestAlphaNumStatsSharing:
    """Tests that AlphaNumP/R/F share the same _AlphaNumStats instance for identical params."""

    @pytest.fixture
    def dataset(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        return dataset

    def test_p_and_r_share_instance(self, dataset):
        p = dataset.metrics.alphanum_p()
        r = dataset.metrics.alphanum_r()
        assert p._alphanum_stats is r._alphanum_stats

    def test_p_and_f_share_instance(self, dataset):
        p = dataset.metrics.alphanum_p()
        f = dataset.metrics.alphanum_f()
        assert p._alphanum_stats is f._alphanum_stats

    def test_all_three_share_instance(self, dataset):
        p = dataset.metrics.alphanum_p()
        r = dataset.metrics.alphanum_r()
        f = dataset.metrics.alphanum_f()
        assert p._alphanum_stats is r._alphanum_stats is f._alphanum_stats

    def test_different_patterns_different_instances(self, dataset):
        a = dataset.metrics.alphanum_p()
        b = dataset.metrics.alphanum_p(pattern=r"[A-Z]{3,}")
        assert a._alphanum_stats is not b._alphanum_stats

    def test_instance_is_cached(self, dataset):
        p1 = dataset.metrics.alphanum_p()
        assert p1._alphanum_stats is p1._alphanum_stats
