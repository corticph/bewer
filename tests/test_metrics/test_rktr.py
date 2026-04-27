"""Tests for bewer.metrics.rktr module."""

import pytest

from bewer import Dataset
from bewer.metrics.rktr import RKTR, RKTR_


class TestRKTRExampleMetric:
    """Tests for RKTR_ (ExampleMetric) class."""

    @pytest.fixture
    def dataset_perfect_match(self):
        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetes",
            key_terms={"key_terms": ["diabetes"]},
        )
        return dataset

    @pytest.fixture
    def dataset_one_char_error(self):
        # "diabetis" vs "diabetes": 1 edit, 8 ref chars → CER = 0.125
        dataset = Dataset()
        dataset.add(
            ref="the patient has diabetes",
            hyp="the patient has diabetis",
            key_terms={"key_terms": ["diabetes"]},
        )
        return dataset

    def test_perfect_match_is_tp(self, dataset_perfect_match):
        example = dataset_perfect_match[0]
        rktr = example.metrics.rktr(vocab="key_terms", threshold=0.0)
        assert rktr.value == 1.0
        assert rktr.num_relaxed_matches == 1

    def test_strict_threshold_rejects_error(self, dataset_one_char_error):
        example = dataset_one_char_error[0]
        rktr = example.metrics.rktr(vocab="key_terms", threshold=0.0)
        assert rktr.value == 0.0
        assert rktr.num_relaxed_matches == 0

    def test_relaxed_threshold_accepts_near_miss(self, dataset_one_char_error):
        # 1/8 CER = 0.125 ≤ 0.2 → TP
        example = dataset_one_char_error[0]
        rktr = example.metrics.rktr(vocab="key_terms", threshold=0.2)
        assert rktr.value == 1.0
        assert rktr.num_relaxed_matches == 1

    def test_threshold_boundary_equal(self, dataset_one_char_error):
        # threshold exactly at CER should be TP (<=)
        example = dataset_one_char_error[0]
        rktr = example.metrics.rktr(vocab="key_terms", threshold=0.125)
        assert rktr.value == 1.0

    def test_threshold_boundary_below(self, dataset_one_char_error):
        # threshold just below CER should be FN
        example = dataset_one_char_error[0]
        rktr = example.metrics.rktr(vocab="key_terms", threshold=0.124)
        assert rktr.value == 0.0

    def test_num_ref_terms(self, dataset_perfect_match):
        example = dataset_perfect_match[0]
        rktr = example.metrics.rktr(vocab="key_terms")
        assert rktr.num_ref_terms == 1

    def test_partial_matches_with_threshold(self):
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes and hypertension",
            hyp="patient has diabetis and hypotension",
            key_terms={"key_terms": ["diabetes", "hypertension"]},
        )
        example = dataset[0]
        # strict: both errors rejected
        rktr_strict = example.metrics.rktr(vocab="key_terms", threshold=0.0)
        assert rktr_strict.num_ref_terms == 2
        assert rktr_strict.num_relaxed_matches == 0
        # relaxed: near-misses accepted (both have 1 edit in ≤13 chars)
        rktr_relaxed = example.metrics.rktr(vocab="key_terms", threshold=0.2)
        assert rktr_relaxed.num_ref_terms == 2
        assert rktr_relaxed.num_relaxed_matches == 2
        assert rktr_relaxed.value == 1.0

    def test_default_threshold_is_zero(self, dataset_perfect_match):
        example = dataset_perfect_match[0]
        rktr_default = example.metrics.rktr(vocab="key_terms")
        rktr_explicit = example.metrics.rktr(vocab="key_terms", threshold=0.0)
        assert rktr_default.value == rktr_explicit.value


class TestRKTRDatasetMetric:
    """Tests for RKTR (dataset-level Metric) class."""

    @pytest.fixture
    def mixed_dataset(self):
        dataset = Dataset()
        dataset.add(
            ref="patient has diabetes",
            hyp="patient has diabetes",
            key_terms={"key_terms": ["diabetes"]},
        )
        dataset.add(
            ref="patient has asthma",
            hyp="patient has astma",
            key_terms={"key_terms": ["asthma"]},
        )
        return dataset

    def test_num_ref_terms_aggregates(self, mixed_dataset):
        rktr = mixed_dataset.metrics.rktr(vocab="key_terms")
        expected = sum(ex.metrics.rktr(vocab="key_terms").num_ref_terms for ex in mixed_dataset)
        assert rktr.num_ref_terms == expected

    def test_num_relaxed_matches_aggregates(self, mixed_dataset):
        rktr = mixed_dataset.metrics.rktr(vocab="key_terms")
        expected = sum(ex.metrics.rktr(vocab="key_terms").num_relaxed_matches for ex in mixed_dataset)
        assert rktr.num_relaxed_matches == expected

    def test_strict_value(self, mixed_dataset):
        rktr = mixed_dataset.metrics.rktr(vocab="key_terms", threshold=0.0)
        assert rktr.num_ref_terms == 2
        assert rktr.num_relaxed_matches == 1
        assert rktr.value == 0.5

    def test_relaxed_accepts_near_miss(self, mixed_dataset):
        # "astma" vs "asthma": 1 edit / 6 chars ≈ 0.167 ≤ 0.2
        rktr = mixed_dataset.metrics.rktr(vocab="key_terms", threshold=0.2)
        assert rktr.value == 1.0

    def test_empty_vocab_raises(self):
        dataset = Dataset()
        with pytest.raises(ValueError, match="not found in dataset key term vocabularies"):
            dataset.metrics.rktr(vocab="key_terms").value

    def test_threshold_above_one_raises(self):
        dataset = Dataset()
        dataset.add(ref="has diabetes", hyp="has diabetes", key_terms={"k": ["diabetes"]})
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            dataset.metrics.rktr(vocab="k", threshold=1.1).value

    def test_threshold_below_zero_raises(self):
        dataset = Dataset()
        dataset.add(ref="has diabetes", hyp="has diabetes", key_terms={"k": ["diabetes"]})
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            dataset.metrics.rktr(vocab="k", threshold=-0.1).value

    def test_all_correct(self):
        dataset = Dataset()
        dataset.add(ref="diabetes mellitus", hyp="diabetes mellitus", key_terms={"k": ["diabetes"]})
        dataset.add(ref="acute asthma", hyp="acute asthma", key_terms={"k": ["asthma"]})
        assert dataset.metrics.rktr(vocab="k").value == 1.0


class TestRKTRPartialPenalty:
    """Tests for the hyp_left_partial / hyp_right_partial boundary penalty."""

    def test_hyp_right_partial_adds_penalty(self):
        # "blood" key term, hyp "bloodpressure": seg[-1].hyp_right_partial=True → +1 edit
        # Levenshtein("blood", "blood") = 0, total = 1, CER = 1/5 = 0.2
        dataset = Dataset()
        dataset.add(
            ref="patient has blood pressure",
            hyp="patient has bloodpressure",
            key_terms={"k": ["blood"]},
        )
        example = dataset[0]
        ts = dataset.metrics._rkt_stats(vocab="k").get_example_metric(example).term_stats
        assert ts[0].char_edits == 1
        assert ts[0].ref_chars == 5
        rktr = example.metrics.rktr(vocab="k", threshold=0.0)
        assert rktr.value == 0.0
        rktr_relaxed = example.metrics.rktr(vocab="k", threshold=0.2)
        assert rktr_relaxed.value == 1.0

    def test_hyp_left_partial_adds_penalty(self):
        # "pressure" key term, hyp "bloodpressure": seg[0].hyp_left_partial=True → +1 edit
        # Levenshtein("pressure", "pressure") = 0, total = 1, CER = 1/8 = 0.125
        dataset = Dataset()
        dataset.add(
            ref="patient has blood pressure",
            hyp="patient has bloodpressure",
            key_terms={"k": ["pressure"]},
        )
        example = dataset[0]
        ts = dataset.metrics._rkt_stats(vocab="k").get_example_metric(example).term_stats
        assert ts[0].char_edits == 1
        assert ts[0].ref_chars == 8
        rktr = example.metrics.rktr(vocab="k", threshold=0.0)
        assert rktr.value == 0.0
        rktr_relaxed = example.metrics.rktr(vocab="k", threshold=0.125)
        assert rktr_relaxed.value == 1.0

    def test_internal_partials_do_not_add_penalty(self):
        # "blood pressure" key term, hyp "bloodpressure": partials are internal to the segment
        # (seg[0].hyp_right_partial and seg[-1].hyp_left_partial), not at the outer boundary.
        # Only Levenshtein contributes (1 edit for missing space), no extra penalty.
        dataset = Dataset()
        dataset.add(
            ref="patient has blood pressure",
            hyp="patient has bloodpressure",
            key_terms={"k": ["blood pressure"]},
        )
        example = dataset[0]
        ts = dataset.metrics._rkt_stats(vocab="k").get_example_metric(example).term_stats
        assert ts[0].char_edits == 1
        assert ts[0].ref_chars == 14


class TestRKTRMetricAttributes:
    """Tests for RKTR metric class attributes."""

    def test_short_name_base(self):
        assert RKTR.short_name_base == "RKTR"

    def test_long_name_base(self):
        assert RKTR.long_name_base == "Relaxed Key Term Recall"

    def test_description(self):
        assert len(RKTR.description) > 0

    def test_example_cls(self):
        assert RKTR.example_cls == RKTR_

    def test_metric_values_main(self):
        assert RKTR.metric_values()["main"] == "value"

    def test_metric_values_other(self):
        values = RKTR.metric_values()
        assert "num_relaxed_matches" in values["other"]
        assert "num_ref_terms" in values["other"]
