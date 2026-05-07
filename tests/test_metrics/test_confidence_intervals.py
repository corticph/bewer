"""Tests for bewer.metrics.confidence (compute_confidence_interval on Metric)."""

import random

import pytest

from bewer import Dataset
from bewer.metrics.confidence import (
    ConfidenceInterval,
    _percentile,
    bootstrap_confidence_interval,
)


@pytest.fixture
def dataset_with_more_variation():
    """A 20-example dataset mixing perfect and erroneous transcriptions, used
    to give the bootstrap distribution non-degenerate variance.
    """
    dataset = Dataset()
    for i in range(10):
        dataset.add(f"the quick brown fox number {i}", f"the quick brown fox number {i}")
    for i in range(10):
        dataset.add(f"hello world example {i}", f"goodbye earth example {i}")
    return dataset


@pytest.fixture
def dataset_with_key_terms():
    """Dataset with a global key term vocabulary, mixing TPs and FNs across
    examples so KT-family metrics produce non-trivial bootstrap distributions.
    """
    dataset = Dataset()
    dataset.add(ref="the fox jumps", hyp="the fox jumps", key_terms={"animals": ["fox"]})
    dataset.add(ref="the fox runs", hyp="the dog runs", key_terms={"animals": ["fox"]})
    dataset.add(ref="a rabbit hops", hyp="a rabbit hops", key_terms={"animals": ["rabbit"]})
    dataset.add(ref="the rabbit eats", hyp="the cat eats", key_terms={"animals": ["rabbit"]})
    dataset.add(ref="fox and rabbit", hyp="fox and hamster", key_terms={"animals": ["fox", "rabbit"]})
    return dataset


class TestConfidenceIntervalDataclass:
    """Tests for the ConfidenceInterval dataclass itself."""

    def test_iter_yields_low_then_high(self):
        ci = ConfidenceInterval(
            low=0.1,
            high=0.3,
            point=0.2,
            level=0.95,
            method="bootstrap",
            n_resamples=1000,
            seed=0,
        )
        low, high = ci
        assert low == 0.1
        assert high == 0.3

    def test_width(self):
        ci = ConfidenceInterval(
            low=0.1,
            high=0.3,
            point=0.2,
            level=0.95,
            method="bootstrap",
            n_resamples=1000,
            seed=0,
        )
        assert ci.width == pytest.approx(0.2)

    def test_is_frozen(self):
        ci = ConfidenceInterval(
            low=0.1,
            high=0.3,
            point=0.2,
            level=0.95,
            method="bootstrap",
            n_resamples=1000,
            seed=0,
        )
        with pytest.raises(Exception):
            ci.low = 0.0


class TestPercentileHelper:
    def test_single_sample(self):
        assert _percentile([1.5], 0.5) == 1.5

    def test_endpoints(self):
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(s, 0.0) == 1.0
        assert _percentile(s, 1.0) == 5.0

    def test_interpolation(self):
        s = [0.0, 10.0]
        assert _percentile(s, 0.5) == pytest.approx(5.0)


class TestBasics:
    def test_returns_confidence_interval(self, sample_dataset):
        ci = sample_dataset.metrics.wer().compute_confidence_interval(seed=0, n_resamples=200)
        assert isinstance(ci, ConfidenceInterval)

    def test_unpacks_as_tuple(self, sample_dataset):
        low, high = sample_dataset.metrics.wer().compute_confidence_interval(seed=0, n_resamples=200)
        assert low <= high

    def test_point_is_inside_interval(self, dataset_with_more_variation):
        wer = dataset_with_more_variation.metrics.wer()
        ci = wer.compute_confidence_interval(seed=0, n_resamples=500)
        assert ci.low <= ci.point <= ci.high

    def test_point_matches_metric_value(self, dataset_with_more_variation):
        wer = dataset_with_more_variation.metrics.wer()
        ci = wer.compute_confidence_interval(seed=0, n_resamples=200)
        assert ci.point == wer.value

    def test_default_level(self, sample_dataset):
        ci = sample_dataset.metrics.wer().compute_confidence_interval(seed=0, n_resamples=200)
        assert ci.level == 0.95

    def test_default_method_is_bootstrap(self, sample_dataset):
        ci = sample_dataset.metrics.wer().compute_confidence_interval(seed=0, n_resamples=200)
        assert ci.method == "bootstrap"

    def test_n_resamples_recorded(self, sample_dataset):
        ci = sample_dataset.metrics.wer().compute_confidence_interval(seed=0, n_resamples=137)
        assert ci.n_resamples == 137

    def test_seed_recorded(self, sample_dataset):
        ci = sample_dataset.metrics.wer().compute_confidence_interval(seed=42, n_resamples=200)
        assert ci.seed == 42


class TestDeterminism:
    def test_same_seed_same_result(self, dataset_with_more_variation):
        m = dataset_with_more_variation.metrics.wer()
        a = m.compute_confidence_interval(seed=7, n_resamples=300)
        b = m.compute_confidence_interval(seed=7, n_resamples=300)
        assert a.low == b.low
        assert a.high == b.high

    def test_different_seeds_produce_varied_results(self, dataset_with_more_variation):
        # The bootstrap should be sensitive to the seed: across several seeds,
        # we expect at least two distinct CIs (a single shared CI across all
        # seeds would indicate the RNG is being ignored).
        m = dataset_with_more_variation.metrics.wer()
        results = {
            (
                m.compute_confidence_interval(seed=s, n_resamples=300).low,
                m.compute_confidence_interval(seed=s, n_resamples=300).high,
            )
            for s in range(8)
        }
        assert len(results) > 1

    def test_seed_none_does_not_crash(self, sample_dataset):
        ci = sample_dataset.metrics.wer().compute_confidence_interval(n_resamples=50)
        assert ci.seed is None


class TestLevelMonotonicity:
    def test_higher_level_wider_interval(self, dataset_with_more_variation):
        m = dataset_with_more_variation.metrics.wer()
        ci_50 = m.compute_confidence_interval(level=0.50, seed=0, n_resamples=500)
        ci_95 = m.compute_confidence_interval(level=0.95, seed=0, n_resamples=500)
        assert ci_95.width >= ci_50.width


class TestParameterValidation:
    def test_level_zero_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="level"):
            sample_dataset.metrics.wer().compute_confidence_interval(level=0.0, n_resamples=10)

    def test_level_one_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="level"):
            sample_dataset.metrics.wer().compute_confidence_interval(level=1.0, n_resamples=10)

    def test_level_above_one_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="level"):
            sample_dataset.metrics.wer().compute_confidence_interval(level=1.5, n_resamples=10)

    def test_level_negative_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="level"):
            sample_dataset.metrics.wer().compute_confidence_interval(level=-0.1, n_resamples=10)

    def test_n_resamples_zero_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="n_resamples"):
            sample_dataset.metrics.wer().compute_confidence_interval(n_resamples=0)

    def test_n_resamples_negative_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="n_resamples"):
            sample_dataset.metrics.wer().compute_confidence_interval(n_resamples=-1)

    def test_unknown_method_raises(self, sample_dataset):
        with pytest.raises(NotImplementedError, match="bagel"):
            sample_dataset.metrics.wer().compute_confidence_interval(method="bagel")


class TestEdgeCases:
    def test_empty_dataset_raises(self, empty_dataset):
        with pytest.raises(ValueError, match="empty"):
            empty_dataset.metrics.wer().compute_confidence_interval(seed=0, n_resamples=10)

    def test_single_example_returns_degenerate_interval(self):
        dataset = Dataset()
        dataset.add("hello world", "hello earth")
        ci = dataset.metrics.wer().compute_confidence_interval(seed=0, n_resamples=10)
        assert ci.low == ci.high == ci.point

    def test_perfect_match_dataset_zero_interval(self, dataset_perfect_match):
        ci = dataset_perfect_match.metrics.wer().compute_confidence_interval(seed=0, n_resamples=200)
        assert ci.low == 0.0
        assert ci.high == 0.0
        assert ci.point == 0.0


class TestMetricsWithoutMainValueRaise:
    """Dataset-level metrics with no @metric_value(main=True) cannot have a CI."""

    def test_error_align_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="no main metric value"):
            sample_dataset.metrics.error_align().compute_confidence_interval(seed=0, n_resamples=10)

    def test_levenshtein_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="no main metric value"):
            sample_dataset.metrics.levenshtein().compute_confidence_interval(seed=0, n_resamples=10)

    def test_summary_raises(self, sample_dataset):
        with pytest.raises(ValueError, match="no main metric value"):
            sample_dataset.metrics.summary().compute_confidence_interval(seed=0, n_resamples=10)


class TestAcrossMetricFamilies:
    def test_wer(self, dataset_with_more_variation):
        ci = dataset_with_more_variation.metrics.wer().compute_confidence_interval(seed=0, n_resamples=200)
        assert ci.low <= ci.point <= ci.high

    def test_cer(self, dataset_with_more_variation):
        ci = dataset_with_more_variation.metrics.cer().compute_confidence_interval(seed=0, n_resamples=200)
        assert ci.low <= ci.point <= ci.high


class TestKeyTermMetricsWithDependencies:
    """KT-family metrics depend on _kt_stats; bootstrap must shadow the dep."""

    def test_ktf(self, dataset_with_key_terms):
        ci = dataset_with_key_terms.metrics.ktf(vocab="animals").compute_confidence_interval(
            seed=0,
            n_resamples=200,
        )
        assert ci.low <= ci.point <= ci.high

    def test_ktp(self, dataset_with_key_terms):
        ci = dataset_with_key_terms.metrics.ktp(vocab="animals").compute_confidence_interval(
            seed=0,
            n_resamples=200,
        )
        assert ci.low <= ci.point <= ci.high

    def test_ktr(self, dataset_with_key_terms):
        ci = dataset_with_key_terms.metrics.ktr(vocab="animals").compute_confidence_interval(
            seed=0,
            n_resamples=200,
        )
        assert ci.low <= ci.point <= ci.high

    def test_kter(self, dataset_with_key_terms):
        ci = dataset_with_key_terms.metrics.kter(vocab="animals").compute_confidence_interval(
            seed=0,
            n_resamples=200,
        )
        assert ci.low <= ci.point <= ci.high

    def test_ktcer(self, dataset_with_key_terms):
        ci = dataset_with_key_terms.metrics.ktcer(vocab="animals").compute_confidence_interval(
            seed=0,
            n_resamples=200,
        )
        assert ci.low <= ci.point <= ci.high

    def test_dependency_metric_not_mutated_by_bootstrap(self, dataset_with_key_terms):
        """Recomputing CI must not corrupt the dependency's _src or cached value."""
        ktf = dataset_with_key_terms.metrics.ktf(vocab="animals")
        kt_stats_before = ktf._kt_stats
        src_before = kt_stats_before._src
        num_tp_before = kt_stats_before.num_tp

        ktf.compute_confidence_interval(seed=0, n_resamples=100)

        assert ktf._kt_stats is kt_stats_before
        assert ktf._kt_stats._src is src_before
        assert ktf._kt_stats.num_tp == num_tp_before


class TestAggregationCorrectness:
    """Sanity check: the bootstrap samples really do reflect the metric's
    aggregation logic, not a naive mean of per-example values.
    """

    def test_bootstrap_reproduces_manual_resample_for_wer(self):
        # Toy dataset: 3 examples with known per-example (num_edits, ref_length).
        dataset = Dataset()
        dataset.add("a b c", "a b c")  # 0 edits, 3 ref
        dataset.add("d e", "x y")  # 2 edits, 2 ref
        dataset.add("f", "f")  # 0 edits, 1 ref

        wer = dataset.metrics.wer()
        n_resamples = 50
        seed = 123

        ci = bootstrap_confidence_interval(wer, level=0.9, n_resamples=n_resamples, seed=seed)

        # Replicate the same RNG sequence and aggregation manually.
        per_example = [(0, 3), (2, 2), (0, 1)]
        rng = random.Random(seed)
        n = 3
        samples = []
        for _ in range(n_resamples):
            sampled = rng.choices(per_example, k=n)
            total_edits = sum(p[0] for p in sampled)
            total_ref = sum(p[1] for p in sampled)
            samples.append(float(total_edits) / total_ref if total_ref > 0 else float(total_edits))
        samples.sort()
        alpha = (1.0 - 0.9) / 2.0
        expected_low = _percentile(samples, alpha)
        expected_high = _percentile(samples, 1.0 - alpha)

        assert ci.low == pytest.approx(expected_low)
        assert ci.high == pytest.approx(expected_high)
