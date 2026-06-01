"""Tests for function-based key term vocabularies (Dataset.add_key_term_function)."""

import re

import pytest

from bewer.core.dataset import Dataset
from bewer.core.key_term import FunctionVocabulary

# Matches alphanumeric terms that mix letters and digits, e.g. "MRI" is not matched (no digit) but
# "HbA1c", "T2", "B12" are. Used across tests as a representative functional vocabulary.
_ALNUM = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+$")


def alphanumeric(tokens, normalized):
    """Return single-token spans for tokens that mix letters and digits."""
    toks = tokens.normalized if normalized else tokens.raw
    return [slice(i, i + 1) for i, tok in enumerate(toks) if _ALNUM.match(tok)]


class TestAddKeyTermFunction:
    """Registration of function-based vocabularies on the dataset."""

    def test_registers_function_vocab(self, empty_dataset):
        empty_dataset.add_key_term_function("alnum", alphanumeric)
        assert "alnum" in empty_dataset._function_vocabs
        assert isinstance(empty_dataset._function_vocabs["alnum"], FunctionVocabulary)
        assert empty_dataset.has_vocab("alnum")

    def test_non_callable_raises(self, empty_dataset):
        with pytest.raises(TypeError, match="must be callable"):
            empty_dataset.add_key_term_function("alnum", "not a function")

    def test_name_collision_with_enumerated_raises(self, empty_dataset):
        empty_dataset.add_key_term_list("terms", ["mri"])
        with pytest.raises(ValueError, match="already exists as an enumerated"):
            empty_dataset.add_key_term_function("terms", alphanumeric)

    def test_enumerated_collision_with_function_raises(self, empty_dataset):
        empty_dataset.add_key_term_function("terms", alphanumeric)
        with pytest.raises(ValueError, match="already exists as a function-based"):
            empty_dataset.add_key_term_list("terms", ["mri"])

    def test_vocab_listed_on_example(self, empty_dataset):
        empty_dataset.add("the HbA1c was high", "the HbA1c was high")
        empty_dataset.add_key_term_function("alnum", alphanumeric)
        assert "alnum" in empty_dataset[0].vocabs


class TestFunctionVocabMatching:
    """Matching behaviour of function-based vocabularies via Text.get_key_term_matches()."""

    def test_matches_alphanumeric_terms(self, empty_dataset):
        empty_dataset.add("patient HbA1c was B12 normal", "patient HbA1c was B12 normal")
        empty_dataset.add_key_term_function("alnum", alphanumeric)
        matches = empty_dataset[0].ref.get_key_term_matches(vocab="alnum")
        matched = sorted(empty_dataset[0].ref.tokens[m].raw[0] for m in matches)
        assert matched == ["B12", "HbA1c"]

    def test_no_matches_returns_empty(self, empty_dataset):
        empty_dataset.add("the quick brown fox", "the quick brown fox")
        empty_dataset.add_key_term_function("alnum", alphanumeric)
        assert empty_dataset[0].ref.get_key_term_matches(vocab="alnum") == []

    def test_unregistered_vocab_returns_empty(self, empty_dataset):
        empty_dataset.add("the quick brown fox", "the quick brown fox")
        assert empty_dataset[0].ref.get_key_term_matches(vocab="unknown") == []

    def test_multi_token_spans(self, empty_dataset):
        empty_dataset.add("magnetic resonance imaging scan", "magnetic resonance imaging scan")

        def first_three_as_term(tokens, normalized):
            return [slice(0, 3)] if len(tokens) >= 3 else []

        empty_dataset.add_key_term_function("phrase", first_three_as_term)
        matches = empty_dataset[0].ref.get_key_term_matches(vocab="phrase")
        assert len(matches) == 1
        assert empty_dataset[0].ref.tokens[matches[0]].raw == ["magnetic", "resonance", "imaging"]

    def test_normalized_flag_is_forwarded(self, empty_dataset):
        """The function receives the request's normalized flag and can match the chosen surface form."""
        empty_dataset.add("Aspirin helps", "Aspirin helps")

        def matches_lowercase_aspirin(tokens, normalized):
            toks = tokens.normalized if normalized else tokens.raw
            return [slice(i, i + 1) for i, t in enumerate(toks) if t == "aspirin"]

        empty_dataset.add_key_term_function("aspirin", matches_lowercase_aspirin)
        # Normalized tokens are lowercased by the default pipeline, so "aspirin" matches.
        assert len(empty_dataset[0].ref.get_key_term_matches(vocab="aspirin", normalized=True)) == 1
        # Raw tokens keep the capital "A", so the same function finds nothing.
        assert empty_dataset[0].ref.get_key_term_matches(vocab="aspirin", normalized=False) == []

    def test_only_local_matches_is_a_noop(self, empty_dataset):
        """A function vocab has no global pool, so only_local_matches is ignored (returns all matches)."""
        empty_dataset.add("patient HbA1c was B12 high", "patient HbA1c was B12 high")
        empty_dataset.add_key_term_function("alnum", alphanumeric)
        ref = empty_dataset[0].ref
        local = ref.get_key_term_matches(vocab="alnum", only_local_matches=True)
        glob = ref.get_key_term_matches(vocab="alnum", only_local_matches=False)
        assert local == glob
        assert len(local) == 2

    def test_invalid_span_type_raises(self, empty_dataset):
        empty_dataset.add("patient HbA1c was high", "patient HbA1c was high")
        empty_dataset.add_key_term_function("bad", lambda tokens, normalized: [(0, 1)])
        with pytest.raises(TypeError, match="must return a list of slices"):
            empty_dataset[0].ref.get_key_term_matches(vocab="bad")

    def test_out_of_range_span_raises(self, empty_dataset):
        empty_dataset.add("patient HbA1c", "patient HbA1c")
        empty_dataset.add_key_term_function("bad", lambda tokens, normalized: [slice(0, 99)])
        with pytest.raises(ValueError, match="invalid span"):
            empty_dataset[0].ref.get_key_term_matches(vocab="bad")


class TestFunctionVocabMetrics:
    """Existing key term metrics operate on function-based vocabularies unchanged."""

    def test_ktr_recall_with_function_vocab(self):
        dataset = Dataset()
        # Two alphanumeric ref terms; the hypothesis transcribes one correctly and corrupts the other.
        dataset.add("patient HbA1c and B12 levels", "patient HbA1c and B13 levels")
        dataset.add_key_term_function("alnum", alphanumeric)
        ktr = dataset.metrics.ktr(vocab="alnum")
        assert ktr.num_ref_terms == 2
        assert ktr.num_matches == 1
        assert ktr.value == pytest.approx(0.5)

    def test_perfect_recall_with_function_vocab(self):
        dataset = Dataset()
        dataset.add("patient HbA1c normal", "patient HbA1c normal")
        dataset.add_key_term_function("alnum", alphanumeric)
        assert dataset.metrics.ktr(vocab="alnum").value == pytest.approx(1.0)

    def test_rktr_with_function_vocab(self):
        """RKTR (and its threshold) operate over a function vocab without raising on validate."""
        dataset = Dataset()
        # One alphanumeric ref term, transcribed with a single-character error (HbA1c -> HbA1d).
        dataset.add("patient HbA1c", "patient HbA1d")
        dataset.add_key_term_function("alnum", alphanumeric)
        # Strict threshold rejects the near-miss; a relaxed threshold accepts it.
        assert dataset.metrics.rktr(vocab="alnum", threshold=0.0).value == pytest.approx(0.0)
        assert dataset.metrics.rktr(vocab="alnum", threshold=0.2).value == pytest.approx(1.0)

    def test_metric_rejects_unknown_vocab(self):
        dataset = Dataset()
        dataset.add("patient HbA1c normal", "patient HbA1c normal")
        with pytest.raises(ValueError, match="not found in dataset key term vocabularies"):
            dataset.metrics.ktr(vocab="unknown").value
