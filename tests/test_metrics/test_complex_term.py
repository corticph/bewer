"""Tests for bewer.metrics.complex_term module (CTR, CTP, CTF)."""

import pytest

from bewer import Dataset
from bewer.metrics.complex_term import CTF, CTP, CTR


class TestComplexTermMetricsAutoVocabulary:
    """The complex-term vocabulary is registered automatically on first use."""

    def test_vocab_registered_on_use(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        assert not dataset.has_vocabulary("complex_terms")
        dataset.metrics.ctr().value
        assert dataset.has_vocabulary("complex_terms")


class TestComplexTermRecall:
    def test_perfect_recall(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 1
        assert ctr.value == 1.0

    def test_missed_complex_term(self):
        dataset = Dataset()
        dataset.add(ref="results showed HbA1c", hyp="results showed HbA1b")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 1
        assert ctr.num_matches == 0
        assert ctr.value == 0.0

    def test_partial_recall_over_dataset(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        dataset.add(ref="results showed HbA1c", hyp="results showed HbA1b")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 2
        assert ctr.num_matches == 1
        assert ctr.value == 0.5

    def test_example_with_no_complex_terms_contributes_nothing(self):
        dataset = Dataset()
        dataset.add(ref="the patient rested well", hyp="the patient rested well")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 0
        assert ctr.value == 0.0

    def test_hyphen_is_scored_strictly(self):
        """A complex term's hyphen is part of its canonical form. Unlike word-level metrics,
        CT metrics use a tokenizer that does not split on hyphens, so a hypothesis writing
        'CT scan' (spaced) does NOT match a reference 'CT-scan'."""
        dataset = Dataset()
        dataset.add(ref="ordered a CT-scan", hyp="ordered a CT scan")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 1  # 'CT-scan'
        assert ctr.num_matches == 0
        assert ctr.value == 0.0

    def test_casing_is_scored_strictly(self):
        """Casing is part of a complex term's identity: the `cased` normalizer preserves case,
        so a hypothesis writing 'mri' does NOT match a reference 'MRI'."""
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an mri")
        ctr = dataset.metrics.ctr()
        assert ctr.num_ref_terms == 1  # 'MRI'
        assert ctr.num_matches == 0
        assert ctr.value == 0.0


class TestComplexTermPrecision:
    def test_perfect_precision(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        assert dataset.metrics.ctp().value == 1.0


class TestComplexTermFScore:
    def test_f1_is_harmonic_mean(self):
        dataset = Dataset()
        dataset.add(
            ref="patient had an MRI and CT-scan showing HbA1c",
            hyp="patient had an MRI and CT-scan showing HbA1b",
        )
        ctr = dataset.metrics.ctr().value
        ctp = dataset.metrics.ctp().value
        ctf = dataset.metrics.ctf().value
        assert ctf == pytest.approx(2 * ctp * ctr / (ctp + ctr))

    def test_beta_must_be_positive(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        with pytest.raises(ValueError, match="beta must be positive"):
            dataset.metrics.ctf(beta=0).value


class TestComplexTermMetricAttributes:
    def test_names(self):
        assert CTR.short_name_base == "CTR"
        assert CTP.short_name_base == "CTP"
        assert CTF.short_name_base == "CTF"
        assert CTR.long_name_base == "Complex Term Recall"

    def test_descriptions_present(self):
        assert len(CTR.description) > 0
        assert len(CTP.description) > 0
        assert len(CTF.description) > 0

    def test_main_value(self):
        assert CTR.metric_values()["main"] == "value"
