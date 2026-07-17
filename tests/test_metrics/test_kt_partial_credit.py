"""Tests for K^[+] partial-credit counts in _KTStats and the public KT metrics."""

import pytest

from bewer import Dataset, Vocabulary


def _dataset(ref: str, hyp: str, terms: list[str], name: str = "vocab") -> Dataset:
    dataset = Dataset()
    dataset.add(ref=ref, hyp=hyp)
    dataset.add_vocabulary(Vocabulary(name=name).add_terms(terms))
    return dataset


class TestPartialCreditStatsExactMatchEquivalence:
    """Single-token terms: K^[+] and K^[=] must give identical counts."""

    def test_tp_single_token_correct(self):
        ds = _dataset("the fox jumps", "the fox jumps", ["fox"])
        exact = ds[0].metrics._kt_stats(vocab="vocab")
        partial = ds[0].metrics._kt_stats(vocab="vocab", partial_credit=True)
        assert partial.num_tp == exact.num_tp == 1
        assert partial.num_fn == exact.num_fn == 0
        assert partial.num_fp == exact.num_fp == 0

    def test_fn_single_token_substitution(self):
        ds = _dataset("the fox jumps", "the dog jumps", ["fox"])
        exact = ds[0].metrics._kt_stats(vocab="vocab")
        partial = ds[0].metrics._kt_stats(vocab="vocab", partial_credit=True)
        assert partial.num_tp == exact.num_tp == 0
        assert partial.num_fn == exact.num_fn == 1
        assert partial.num_fp == exact.num_fp == 0

    def test_fp_single_token_insertion(self):
        # hyp inserts a vocab term not present in ref
        ds = _dataset("the jumps", "the fox jumps", ["fox"])
        exact = ds[0].metrics._kt_stats(vocab="vocab")
        partial = ds[0].metrics._kt_stats(vocab="vocab", partial_credit=True)
        assert partial.num_tp == exact.num_tp == 0
        assert partial.num_fn == exact.num_fn == 0
        assert partial.num_fp == exact.num_fp == 1


class TestPartialCreditMultiToken:
    """Multi-token terms: K^[+] counts token positions, K^[=] counts occurrences."""

    def test_correct_multitoken_tp(self):
        # K^[=]: 1 TP occurrence; K^[+]: 2 TP positions (both tokens matched)
        ds = _dataset("blood sugar levels", "blood sugar levels", ["blood sugar"])
        exact = ds[0].metrics._kt_stats(vocab="vocab")
        partial = ds[0].metrics._kt_stats(vocab="vocab", partial_credit=True)
        assert exact.num_tp == 1
        assert partial.num_tp == 2
        assert partial.num_fn == 0
        assert partial.num_fp == 0

    def test_one_token_wrong(self):
        # ref="blood sugar", hyp="blood sweet" — second token substituted
        # K^[=]: 0 TP, 1 FN (whole term fails)
        # K^[+]: 1 TP (blood), 1 FN (sugar→sweet)
        ds = _dataset("blood sugar", "blood sweet", ["blood sugar"])
        exact = ds[0].metrics._kt_stats(vocab="vocab")
        partial = ds[0].metrics._kt_stats(vocab="vocab", partial_credit=True)
        assert exact.num_tp == 0
        assert exact.num_fn == 1
        assert partial.num_tp == 1
        assert partial.num_fn == 1
        assert partial.num_fp == 0

    def test_internal_insertion(self):
        # ref="blood sugar", hyp="blood sweet sugar" — insertion between the two correct tokens
        # K^[=]: 0 TP, 1 FN (whole term span contains an insertion)
        # K^[+]: 2 TP (both constituent tokens matched), insertion not penalised per-token
        ds = _dataset("blood sugar", "blood sweet sugar", ["blood sugar"])
        exact = ds[0].metrics._kt_stats(vocab="vocab")
        partial = ds[0].metrics._kt_stats(vocab="vocab", partial_credit=True)
        assert exact.num_tp == 0
        assert exact.num_fn == 1
        assert partial.num_tp == 2
        assert partial.num_fn == 0
        assert partial.num_fp == 0

    def test_fp_multitoken_insertion(self):
        # ref has no "blood sugar"; hyp inserts it
        # K^[=]: 1 FP occurrence; K^[+]: 2 FP positions (both inserted tokens inside I_H)
        ds = _dataset("the levels", "the blood sugar levels", ["blood sugar"])
        exact = ds[0].metrics._kt_stats(vocab="vocab")
        partial = ds[0].metrics._kt_stats(vocab="vocab", partial_credit=True)
        assert exact.num_fp == 1
        assert partial.num_fp == 2
        assert partial.num_tp == 0
        assert partial.num_fn == 0


class TestPartialCreditPositionDeduplication:
    """Positions covered by multiple overlapping terms are counted only once in K^[+]."""

    def test_overlapping_terms_count_once(self):
        # vocab: ["blood", "blood sugar"]
        # ref="blood sugar", hyp="blood sweet"
        # I_R = {0, 1} (pos 0 from "blood" AND "blood sugar"; pos 1 from "blood sugar")
        # Alignment: MATCH(blood), SUBSTITUTE(sugar→sweet)
        # K^[+]: TP=1 (pos 0 matched), FN=1 (pos 1 not matched), FP=0 (hyp "blood" matched so not FP)
        ds = _dataset("blood sugar", "blood sweet", ["blood", "blood sugar"])
        partial = ds[0].metrics._kt_stats(vocab="vocab", partial_credit=True)
        assert partial.num_tp == 1
        assert partial.num_fn == 1
        assert partial.num_fp == 0


class TestPartialCreditPublicMetrics:
    """KTR, KTP, KTF, KTER with partial_credit=True produce position-level values."""

    @pytest.fixture
    def ds_one_wrong(self):
        # ref="blood sugar", hyp="blood sweet"
        # K^[+]: TP=1, FN=1, FP=0
        ds = Dataset()
        ds.add(ref="blood sugar", hyp="blood sweet")
        ds.add_vocabulary(Vocabulary(name="terms").add_terms(["blood sugar"]))
        return ds

    def test_ktr_partial_credit(self, ds_one_wrong):
        assert ds_one_wrong.metrics.ktr(vocab="terms", partial_credit=True).value == pytest.approx(0.5)

    def test_ktr_exact_match(self, ds_one_wrong):
        assert ds_one_wrong.metrics.ktr(vocab="terms", partial_credit=False).value == pytest.approx(0.0)

    def test_ktp_partial_credit(self, ds_one_wrong):
        # TP=1, FP=0 → precision=1.0
        assert ds_one_wrong.metrics.ktp(vocab="terms", partial_credit=True).value == pytest.approx(1.0)

    def test_ktp_exact_match(self, ds_one_wrong):
        # TP=0, FP=0 → 0.0 (zero denominator)
        assert ds_one_wrong.metrics.ktp(vocab="terms", partial_credit=False).value == pytest.approx(0.0)

    def test_ktf_partial_credit(self, ds_one_wrong):
        # TP=1, FN=1, FP=0, beta=1 → 2*1/(2*1 + 1*1 + 0) = 2/3
        assert ds_one_wrong.metrics.ktf(vocab="terms", partial_credit=True).value == pytest.approx(2 / 3)

    def test_ktf_exact_match(self, ds_one_wrong):
        assert ds_one_wrong.metrics.ktf(vocab="terms", partial_credit=False).value == pytest.approx(0.0)

    def test_kter_partial_credit(self, ds_one_wrong):
        # FN=1, TP+FN=2 → KTER=0.5
        assert ds_one_wrong.metrics.kter(vocab="terms", partial_credit=True).value == pytest.approx(0.5)

    def test_kter_exact_match(self, ds_one_wrong):
        # FN=1, TP+FN=1 → KTER=1.0
        assert ds_one_wrong.metrics.kter(vocab="terms", partial_credit=False).value == pytest.approx(1.0)


class TestPartialCreditKterDenominator:
    """KTER denominator uses TP+FN in both modes — backward-compatible with K^[=]."""

    def test_kter_denominator_unchanged_for_exact_match(self):
        # K^[=]: TP+FN == num_ref_terms always, so the formula change is invisible
        ds = Dataset()
        ds.add(ref="the fox jumps", hyp="the dog jumps")
        ds.add_vocabulary(Vocabulary(name="v").add_terms(["fox"]))
        kter = ds.metrics.kter(vocab="v")
        # TP=0, FN=1 → KTER = 1/(0+1) = 1.0
        assert kter.value == pytest.approx(1.0)

    def test_kter_zero_ref_terms(self):
        ds = Dataset()
        ds.add(ref="the jumps", hyp="the jumps")
        ds.add_vocabulary(Vocabulary(name="v").add_terms(["fox"]))
        kter = ds.metrics.kter(vocab="v")
        assert kter.value == pytest.approx(0.0)


class TestPartialCreditDatasetLevel:
    """Dataset-level aggregation works correctly for partial_credit=True."""

    def test_dataset_aggregation(self):
        ds = Dataset()
        ds.add(ref="blood sugar levels", hyp="blood sugar levels")  # K^[+]: TP=2
        ds.add(ref="blood sugar levels", hyp="blood sweet levels")  # K^[+]: TP=1, FN=1
        ds.add_vocabulary(Vocabulary(name="terms").add_terms(["blood sugar"]))

        stats = ds.metrics._kt_stats(vocab="terms", partial_credit=True)
        assert stats.num_tp == 3
        assert stats.num_fn == 1
        assert stats.num_fp == 0

    def test_ktr_dataset_aggregation(self):
        ds = Dataset()
        ds.add(ref="blood sugar levels", hyp="blood sugar levels")  # TP=2, FN=0
        ds.add(ref="blood sugar levels", hyp="blood sweet levels")  # TP=1, FN=1
        ds.add_vocabulary(Vocabulary(name="terms").add_terms(["blood sugar"]))

        ktr = ds.metrics.ktr(vocab="terms", partial_credit=True)
        # total TP=3, total FN=1 → recall = 3/4
        assert ktr.value == pytest.approx(3 / 4)
