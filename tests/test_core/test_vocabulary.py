"""Tests for the standalone Vocabulary object (builder, extractors, sharing, caching)."""

import gc

import pytest

from bewer import (
    Dataset,
    DatasetFrozenError,
    Vocabulary,
    VocabularyExtractorError,
    VocabularyFrozenError,
)


class TestVocabularyBuilder:
    """Tests for the Vocabulary construction / builder API."""

    def test_name_required_and_exposed(self):
        vocab = Vocabulary(name="animals")
        assert vocab.name == "animals"

    def test_non_string_name_raises(self):
        with pytest.raises(TypeError):
            Vocabulary(name=123)

    def test_builder_methods_chain_and_return_self(self):
        vocab = Vocabulary(name="x")
        assert vocab.add_terms(["a"]) is vocab
        assert vocab.add_extractor(lambda ds: []) is vocab

    def test_has_sources(self):
        assert Vocabulary(name="x").has_sources is False
        assert Vocabulary(name="x").add_terms(["a"]).has_sources is True
        assert Vocabulary(name="x").add_extractor(lambda ds: []).has_sources is True

    def test_add_terms_rejects_bare_string(self):
        with pytest.raises(TypeError):
            Vocabulary(name="x").add_terms("not_a_list")

    def test_add_terms_rejects_non_iterable(self):
        with pytest.raises(TypeError):
            Vocabulary(name="x").add_terms(42)

    def test_add_terms_rejects_non_string_element(self):
        with pytest.raises(TypeError):
            Vocabulary(name="x").add_terms(["ok", 5])

    def test_add_extractor_rejects_non_callable(self):
        with pytest.raises(TypeError):
            Vocabulary(name="x").add_extractor("not callable")

    def test_add_file(self, tmp_path):
        path = tmp_path / "terms.txt"
        path.write_text("diabetes\nasthma\n")
        vocab = Vocabulary(name="m").add_file(str(path))
        assert vocab.has_sources is True

    def test_add_file_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            Vocabulary(name="m").add_file("/nonexistent/terms.txt")


class TestVocabularyMatching:
    """End-to-end matching through metrics with the new Vocabulary API."""

    def test_static_vocab_matches(self):
        dataset = Dataset()
        dataset.add("the patient has diabetes", "the patient has diabetes")
        dataset.add_vocabulary(Vocabulary(name="m").add_terms(["diabetes"]))
        assert dataset.metrics.ktr(vocab="m").value == 1.0

    def test_extractor_from_refs_matched_like_static(self):
        """A term produced by an extractor reading refs is matched exactly like a static term."""
        static = Dataset()
        static.add("the patient has diabetes", "the patient has diabetes")
        static.add_vocabulary(Vocabulary(name="m").add_terms(["diabetes"]))

        extracted = Dataset()
        extracted.add("the patient has diabetes", "the patient has diabetes")
        extracted.add_vocabulary(Vocabulary(name="m").add_extractor(lambda ds: [e.ref.raw.split()[-1] for e in ds]))

        assert extracted.metrics.ktr(vocab="m").value == static.metrics.ktr(vocab="m").value == 1.0

    def test_extractor_can_read_hyps(self):
        """Extractors may pull terms from hypotheses; such terms match like any other."""
        dataset = Dataset()
        dataset.add("the cold weather today", "a cold morning")
        # "cold" is extracted from the hypothesis and also appears in the reference.
        dataset.add_vocabulary(Vocabulary(name="h").add_extractor(lambda ds: [ex.hyp.tokens.raw[1] for ex in ds]))
        kt = dataset.metrics._kt_stats(vocab="h")
        assert kt.num_ref_terms == 1

    def test_multi_source_union_and_dedup(self):
        """Static + extractor sources union; duplicates collapse by raw string."""
        dataset = Dataset()
        dataset.add("diabetes and asthma", "diabetes and asthma")
        vocab = (
            Vocabulary(name="m")
            .add_terms(["diabetes"])
            .add_extractor(lambda ds: ["diabetes", "asthma"])  # "diabetes" dups the static term
        )
        dataset.add_vocabulary(vocab)
        # Two distinct ref terms (diabetes, asthma), not three.
        assert dataset.metrics._kt_stats(vocab="m").num_ref_terms == 2


class TestVocabularySharing:
    """A single Vocabulary attached to multiple datasets resolves per dataset."""

    def test_extractor_resolves_per_dataset(self):
        vocab = Vocabulary(name="last").add_extractor(lambda ds: [e.ref.raw.split()[-1] for e in ds])

        d1 = Dataset()
        d1.add("the patient has diabetes", "the patient has diabetes")
        d1.add_vocabulary(vocab)

        d2 = Dataset()
        d2.add("severe asthma", "severe asthma")
        d2.add_vocabulary(vocab)

        # Same Vocabulary object, different resolved terms / matches per dataset.
        assert d1.metrics._kt_stats(vocab="last").num_ref_terms == 1
        assert d2.metrics._kt_stats(vocab="last").num_ref_terms == 1
        # The resolved term sets differ between the two datasets.
        terms1 = {kt.raw for kt in vocab._cache[d1][next(iter(vocab._cache[d1]))].terms}
        terms2 = {kt.raw for kt in vocab._cache[d2][next(iter(vocab._cache[d2]))].terms}
        assert terms1 == {"diabetes"}
        assert terms2 == {"asthma"}

    def test_add_vocabulary_name_collision_with_different_object_raises(self):
        dataset = Dataset()
        dataset.add("a b", "a b")
        dataset.add_vocabulary(Vocabulary(name="m").add_terms(["a"]))
        with pytest.raises(ValueError):
            dataset.add_vocabulary(Vocabulary(name="m").add_terms(["b"]))

    def test_add_vocabulary_same_object_idempotent(self):
        dataset = Dataset()
        dataset.add("a b", "a b")
        vocab = Vocabulary(name="m").add_terms(["a"])
        dataset.add_vocabulary(vocab)
        dataset.add_vocabulary(vocab)  # no error
        assert dataset._vocabularies["m"] is vocab

    def test_add_vocabulary_rejects_non_vocabulary(self):
        dataset = Dataset()
        dataset.add("a b", "a b")
        with pytest.raises(TypeError):
            dataset.add_vocabulary("not a vocabulary")


class TestVocabularyFreezeOnRegistration:
    """Attaching a vocabulary freezes its definition (loud, not silent)."""

    def test_add_terms_after_registration_raises(self):
        dataset = Dataset()
        dataset.add("a b", "a b")
        vocab = Vocabulary(name="m").add_terms(["a"])
        dataset.add_vocabulary(vocab)
        with pytest.raises(VocabularyFrozenError):
            vocab.add_terms(["b"])

    def test_add_extractor_after_registration_raises(self):
        dataset = Dataset()
        dataset.add("a b", "a b")
        vocab = Vocabulary(name="m").add_terms(["a"])
        dataset.add_vocabulary(vocab)
        with pytest.raises(VocabularyFrozenError):
            vocab.add_extractor(lambda ds: ["b"])

    def test_add_file_after_registration_raises(self, tmp_path):
        path = tmp_path / "terms.txt"
        path.write_text("b\n")
        dataset = Dataset()
        dataset.add("a b", "a b")
        vocab = Vocabulary(name="m").add_terms(["a"])
        dataset.add_vocabulary(vocab)
        with pytest.raises(VocabularyFrozenError):
            vocab.add_file(str(path))

    def test_can_be_attached_to_multiple_datasets_after_freeze(self):
        """Freezing blocks definition edits, not re-attachment: sharing still works."""
        vocab = Vocabulary(name="m").add_terms(["fox"])
        d1 = Dataset()
        d1.add("the quick brown fox", "the quick brown fox")
        d1.add_vocabulary(vocab)  # freezes
        d2 = Dataset()
        d2.add("a fox", "a fox")
        d2.add_vocabulary(vocab)  # already frozen, still allowed
        assert d1.metrics.ktr(vocab="m").value == 1.0
        assert d2.metrics.ktr(vocab="m").value == 1.0


class TestVocabularyClone:
    """clone() shares Vocabulary objects and re-resolves them against the clone."""

    def test_clone_shares_vocab_and_recomputes(self):
        dataset = Dataset()
        dataset.add("the patient has diabetes", "the patient has diabetis")
        dataset.add_vocabulary(Vocabulary(name="m").add_extractor(lambda ds: ["diabetes"]))
        clone = dataset.clone()
        # Same Vocabulary object is shared.
        assert clone._vocabularies["m"] is dataset._vocabularies["m"]
        # The clone resolves and computes independently.
        assert clone.metrics.ktr(vocab="m").value == 0.0  # "diabetis" != "diabetes"


class TestVocabularyFreezeAndCache:
    """Resolving a vocabulary freezes the dataset (closes the stale-cache issue)."""

    def test_matching_freezes_dataset(self):
        dataset = Dataset()
        dataset.add("the quick brown fox", "the quick brown fox")
        dataset.add_vocabulary(Vocabulary(name="a").add_terms(["fox"]))
        assert dataset.is_frozen is False
        # Direct matching triggers resolution, which freezes the dataset.
        dataset[0].ref.get_key_term_matches(vocab="a")
        assert dataset.is_frozen is True
        with pytest.raises(DatasetFrozenError):
            dataset.add("more", "more")

    def test_shared_vocab_cache_is_released_on_dataset_gc(self):
        vocab = Vocabulary(name="a").add_terms(["fox"])

        def use_once():
            d = Dataset()
            d.add("the quick brown fox", "the quick brown fox")
            d.add_vocabulary(vocab)
            d.metrics.ktr(vocab="a").value  # triggers resolution -> cache entry keyed by d
            assert len(vocab._cache) == 1

        use_once()
        gc.collect()
        # The dataset is unreferenced; its weak cache entry is gone — no leak for a shared vocab.
        assert len(vocab._cache) == 0


class TestVocabularyEdgeCases:
    def test_empty_extractor_output(self):
        dataset = Dataset()
        dataset.add("a b c", "a b c")
        dataset.add_vocabulary(Vocabulary(name="e").add_extractor(lambda ds: []))
        assert dataset[0].ref.get_key_term_matches(vocab="e") == []

    def test_extractor_returning_none(self):
        dataset = Dataset()
        dataset.add("a b c", "a b c")
        dataset.add_vocabulary(Vocabulary(name="e").add_extractor(lambda ds: None))
        assert dataset[0].ref.get_key_term_matches(vocab="e") == []

    def test_extractor_raising_wrapped(self):
        def boom(ds):
            raise RuntimeError("nope")

        dataset = Dataset()
        dataset.add("a b c", "a b c")
        dataset.add_vocabulary(Vocabulary(name="e").add_extractor(boom))
        with pytest.raises(VocabularyExtractorError):
            dataset.metrics.ktr(vocab="e").value

    def test_extractor_non_string_output_raises(self):
        dataset = Dataset()
        dataset.add("a b c", "a b c")
        dataset.add_vocabulary(Vocabulary(name="e").add_extractor(lambda ds: [1, 2]))
        with pytest.raises(TypeError):
            dataset.metrics.ktr(vocab="e").value

    def test_source_less_vocab_yields_no_matches(self):
        dataset = Dataset()
        dataset.add("a b c", "a b c")
        dataset.add_vocabulary(Vocabulary(name="empty"))
        assert dataset[0].ref.get_key_term_matches(vocab="empty") == []
        assert dataset.metrics._kt_stats(vocab="empty").num_ref_terms == 0

    def test_unregistered_vocab_name_returns_no_matches(self):
        dataset = Dataset()
        dataset.add("a b c", "a b c")
        assert dataset[0].ref.get_key_term_matches(vocab="missing") == []
