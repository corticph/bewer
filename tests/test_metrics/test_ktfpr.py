"""Tests for bewer.metrics.ktfpr module."""

import pytest

from bewer import Dataset, Vocabulary
from bewer.metrics.ktfpr import KTFPR


def _dataset(ref: str, hyp: str, terms: list[str], name: str = "vocab") -> Dataset:
    dataset = Dataset()
    dataset.add(ref=ref, hyp=hyp)
    dataset.add_vocabulary(Vocabulary(name=name).add_terms(terms))
    return dataset


class TestKTFPRValidation:
    """KTFPR is only defined for partial_credit=True."""

    def test_partial_credit_false_raises(self):
        ds = _dataset("the fox jumps", "the fox jumps", ["fox"])
        with pytest.raises(ValueError, match="partial_credit=True"):
            ds.metrics.ktfpr(vocab="vocab", partial_credit=False).value

    def test_unknown_vocab_raises(self):
        ds = Dataset()
        ds.add(ref="hello", hyp="hello")
        with pytest.raises(ValueError, match="not found in dataset key term vocabularies"):
            ds.metrics.ktfpr(vocab="missing").value

    def test_default_partial_credit_is_true(self):
        ds = _dataset("the fox jumps", "the fox jumps", ["fox"])
        # should not raise
        assert ds.metrics.ktfpr(vocab="vocab").value == pytest.approx(0.0)


class TestKTFPRNoFalsePositives:
    """FPR is 0 when the hypothesis contains no spurious key term detections."""

    def test_correct_transcription(self):
        ds = _dataset("the fox jumps", "the fox jumps", ["fox"])
        assert ds.metrics.ktfpr(vocab="vocab").value == pytest.approx(0.0)

    def test_fn_only_no_fp(self):
        # key term missed entirely — FP=0, FPR=0
        ds = _dataset("the fox jumps", "the dog jumps", ["fox"])
        assert ds.metrics.ktfpr(vocab="vocab").value == pytest.approx(0.0)


class TestKTFPRWithFalsePositives:
    """FPR > 0 when hypothesis inserts spurious key term tokens."""

    def test_single_token_insertion(self):
        # ref="the jumps" (2 tokens), hyp="the fox jumps" — fox inserted
        # FP=1, N_R=2 → FPR=0.5
        ds = _dataset("the jumps", "the fox jumps", ["fox"])
        fpr = ds.metrics.ktfpr(vocab="vocab")
        assert fpr.num_fp == 1
        assert fpr.num_ref_tokens == 2
        assert fpr.value == pytest.approx(0.5)

    def test_multitoken_insertion(self):
        # ref="the levels" (2 tokens), hyp="the blood sugar levels" — 2-token term inserted
        # K^[+] FP=2, N_R=2 → FPR=1.0
        ds = _dataset("the levels", "the blood sugar levels", ["blood sugar"])
        fpr = ds.metrics.ktfpr(vocab="vocab")
        assert fpr.num_fp == 2
        assert fpr.num_ref_tokens == 2
        assert fpr.value == pytest.approx(1.0)

    def test_fp_correctly_transcribed_not_counted(self):
        # Only mismatched hyp positions count as FP.
        # ref="the fox jumps", hyp="the fox fox jumps" — second fox is an insertion → FP=1
        ds = _dataset("the fox jumps", "the fox fox jumps", ["fox"])
        fpr = ds.metrics.ktfpr(vocab="vocab")
        assert fpr.num_fp == 1


class TestKTFPRNumRefTokens:
    """num_ref_tokens equals the reference token count, vocabulary-independent."""

    def test_ref_token_count(self):
        ds = _dataset("the quick brown fox jumps", "the quick brown fox jumps", ["fox"])
        assert ds.metrics.ktfpr(vocab="vocab").num_ref_tokens == 5

    def test_ref_token_count_independent_of_vocab(self):
        ds = Dataset()
        ds.add(ref="the quick brown fox jumps", hyp="the quick brown fox jumps")
        ds.add_vocabulary(Vocabulary(name="v1").add_terms(["fox"]))
        ds.add_vocabulary(Vocabulary(name="v2").add_terms(["quick", "brown", "fox", "jumps"]))
        assert ds.metrics.ktfpr(vocab="v1").num_ref_tokens == ds.metrics.ktfpr(vocab="v2").num_ref_tokens


class TestKTFPRDatasetLevel:
    """Dataset-level FPR aggregates across examples."""

    def test_dataset_aggregation(self):
        ds = Dataset()
        ds.add(ref="the jumps", hyp="the fox jumps")  # FP=1, N_R=2
        ds.add(ref="the run", hyp="the run")  # FP=0, N_R=2
        ds.add_vocabulary(Vocabulary(name="v").add_terms(["fox"]))
        fpr = ds.metrics.ktfpr(vocab="v")
        assert fpr.num_fp == 1
        assert fpr.num_ref_tokens == 4
        assert fpr.value == pytest.approx(0.25)

    def test_zero_ref_tokens(self):
        ds = Dataset()
        ds.add(ref="", hyp="")
        ds.add_vocabulary(Vocabulary(name="v").add_terms(["fox"]))
        assert ds.metrics.ktfpr(vocab="v").value == pytest.approx(0.0)


class TestKTFPRExampleLevel:
    """Example-level KTFPR_ properties."""

    def test_example_num_fp(self):
        ds = _dataset("the jumps", "the fox jumps", ["fox"])
        assert ds[0].metrics.ktfpr(vocab="vocab").num_fp == 1

    def test_example_num_ref_tokens(self):
        ds = _dataset("the jumps", "the fox jumps", ["fox"])
        assert ds[0].metrics.ktfpr(vocab="vocab").num_ref_tokens == 2

    def test_example_value(self):
        ds = _dataset("the jumps", "the fox jumps", ["fox"])
        assert ds[0].metrics.ktfpr(vocab="vocab").value == pytest.approx(0.5)


class TestKTFPRSharing:
    """KTFPR shares its _KTStats instance with KTR/KTP/KTF when params match."""

    def test_ktfpr_shares_kt_stats_with_ktr(self):
        ds = _dataset("the fox jumps", "the fox jumps", ["fox"])
        ktfpr = ds.metrics.ktfpr(vocab="vocab")
        ktr = ds.metrics.ktr(vocab="vocab", partial_credit=True)
        assert ktfpr._kt_stats is ktr._kt_stats


class TestKTFPRRegistered:
    """KTFPR is registered in the metric registry."""

    def test_ktfpr_accessible_via_metrics_accessor(self):
        ds = _dataset("the fox jumps", "the fox jumps", ["fox"])
        result = ds.metrics.ktfpr(vocab="vocab")
        assert isinstance(result, KTFPR)
