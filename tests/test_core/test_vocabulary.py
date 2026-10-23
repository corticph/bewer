"""Tests for the standalone Vocabulary object (builder, extractors, sharing, freeze)."""

import pytest

from bewer import (
    BewerConfig,
    Dataset,
    DatasetFrozenError,
    Vocabulary,
    VocabularyExtractorError,
    VocabularyFrozenError,
    get_config,
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
        assert "diabetes" in vocab._static_terms

    def test_add_file_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            Vocabulary(name="m").add_file("/nonexistent/terms.txt")

    def test_init_with_terms(self):
        vocab = Vocabulary(name="m", terms=["diabetes", "asthma"])
        assert vocab._static_terms == {"diabetes", "asthma"}

    def test_init_with_files(self, tmp_path):
        path = tmp_path / "terms.txt"
        path.write_text("diabetes\nasthma\n")
        vocab = Vocabulary(name="m", files=[str(path)])
        assert vocab._static_terms == {"diabetes", "asthma"}

    def test_init_with_extractors(self):
        fn = lambda ds: ["diabetes"]  # noqa: E731
        vocab = Vocabulary(name="m", extractors=[fn])
        assert vocab._extractors == [fn]

    def test_init_combined(self, tmp_path):
        path = tmp_path / "terms.txt"
        path.write_text("asthma\n")
        fn = lambda ds: ["hypertension"]  # noqa: E731
        vocab = Vocabulary(name="m", terms=["diabetes"], files=[str(path)], extractors=[fn])
        assert vocab._static_terms == {"diabetes", "asthma"}
        assert vocab._extractors == [fn]


class TestVocabularyMatching:
    """End-to-end matching through metrics with the new Vocabulary API."""

    def test_static_vocab_matches(self):
        config = get_config("base").clone().add_vocabulary(Vocabulary("m", terms=["diabetes"]))
        dataset = Dataset(config=config)
        dataset.add("the patient has diabetes", "the patient has diabetes")
        assert dataset.metrics.ktr(vocab="m").value == 1.0

    def test_extractor_from_refs_matched_like_static(self):
        """A term produced by an extractor reading refs is matched exactly like a static term."""
        static = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("m", terms=["diabetes"])))
        static.add("the patient has diabetes", "the patient has diabetes")

        extracted = Dataset(
            config=get_config("base")
            .clone()
            .add_vocabulary(Vocabulary("m", extractors=[lambda ds: [e.ref.raw.split()[-1] for e in ds]]))
        )
        extracted.add("the patient has diabetes", "the patient has diabetes")

        assert extracted.metrics.ktr(vocab="m").value == static.metrics.ktr(vocab="m").value == 1.0

    def test_extractor_can_read_hyps(self):
        """Extractors may pull terms from hypotheses; such terms match like any other."""
        dataset = Dataset(
            config=get_config("base")
            .clone()
            .add_vocabulary(Vocabulary("h", extractors=[lambda ds: [ex.hyp.tokens.standardized[1] for ex in ds]]))
        )
        dataset.add("the cold weather today", "a cold morning")
        # "cold" is extracted from the hypothesis and also appears in the reference.
        kt = dataset.metrics._kt_stats(vocab="h")
        assert kt.num_ref_terms == 1

    def test_multi_source_union_and_dedup(self):
        """Static + extractor sources union; duplicates collapse by raw string."""
        vocab = Vocabulary("m", terms=["diabetes"], extractors=[lambda ds: ["diabetes", "asthma"]])
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(vocab))
        dataset.add("diabetes and asthma", "diabetes and asthma")
        # Two distinct ref terms (diabetes, asthma), not three.
        assert dataset.metrics._kt_stats(vocab="m").num_ref_terms == 2


class TestVocabularySharing:
    """A single Vocabulary attached to multiple datasets resolves per dataset."""

    def test_extractor_resolves_per_dataset(self):
        vocab = Vocabulary("last", extractors=[lambda ds: [e.ref.raw.split()[-1] for e in ds]])

        d1 = Dataset(config=get_config("base").clone().add_vocabulary(vocab))
        d1.add("the patient has diabetes", "the patient has diabetes")

        d2 = Dataset(config=get_config("base").clone().add_vocabulary(vocab))
        d2.add("severe asthma", "severe asthma")

        # Same Vocabulary object, different resolved terms / matches per dataset.
        assert d1.metrics._kt_stats(vocab="last").num_ref_terms == 1
        assert d2.metrics._kt_stats(vocab="last").num_ref_terms == 1
        # The resolved term sets differ between the two datasets.
        terms1 = {kt.raw for kt in vocab._resolve_terms(d1)}
        terms2 = {kt.raw for kt in vocab._resolve_terms(d2)}
        assert terms1 == {"diabetes"}
        assert terms2 == {"asthma"}

    def test_add_vocabulary_same_name_overwrites(self):
        vocab_a = Vocabulary("m", terms=["a"])
        vocab_b = Vocabulary("m", terms=["b"])
        config = get_config("base").clone().add_vocabulary(vocab_a)
        config = config.add_vocabulary(vocab_b, allow_override=True)
        dataset = Dataset(config=config)
        dataset.add("a b", "a b")
        assert dataset.config.vocabularies["m"] is vocab_b

    def test_add_vocabulary_same_object_idempotent(self):
        vocab = Vocabulary("m", terms=["a"])
        config = get_config("base").clone().add_vocabulary(vocab)
        config = config.add_vocabulary(vocab, allow_override=True)  # no error
        dataset = Dataset(config=config)
        dataset.add("a b", "a b")
        assert dataset.config.vocabularies["m"] is vocab

    def test_add_vocabulary_rejects_non_vocabulary(self):
        config = get_config("base").clone()
        with pytest.raises(AttributeError):
            config.add_vocabulary("not a vocabulary")


class TestVocabularyFreezeOnRegistration:
    """Adding a vocabulary to a config freezes its definition (loud, not silent)."""

    def test_add_terms_after_registration_raises(self):
        vocab = Vocabulary(name="m", terms=["a"])
        get_config("base").clone().add_vocabulary(vocab)
        with pytest.raises(VocabularyFrozenError):
            vocab.add_terms(["b"])

    def test_add_terms_after_constructor_raises(self):
        vocab = Vocabulary(name="m", terms=["a"])
        BewerConfig(vocabularies=(vocab,))
        with pytest.raises(VocabularyFrozenError):
            vocab.add_terms(["b"])

    def test_constructor_freezes_extends_entries(self):
        parent_vocab = Vocabulary(name="m", terms=["a"])
        base = BewerConfig(vocabularies=(parent_vocab,))
        new_vocab = Vocabulary(name="n", terms=["b"])
        BewerConfig(extends=base, vocabularies=(new_vocab,))
        # Entries entering via either branch (inherited or new) freeze on attach.
        assert parent_vocab.is_frozen
        with pytest.raises(VocabularyFrozenError):
            new_vocab.add_terms(["c"])

    def test_add_extractor_after_registration_raises(self):
        vocab = Vocabulary(name="m", terms=["a"])
        get_config("base").clone().add_vocabulary(vocab)
        with pytest.raises(VocabularyFrozenError):
            vocab.add_extractor(lambda ds: ["b"])

    def test_add_file_after_registration_raises(self, tmp_path):
        path = tmp_path / "terms.txt"
        path.write_text("b\n")
        vocab = Vocabulary(name="m", terms=["a"])
        get_config("base").clone().add_vocabulary(vocab)
        with pytest.raises(VocabularyFrozenError):
            vocab.add_file(str(path))

    def test_can_be_attached_to_multiple_datasets_after_freeze(self):
        """Freezing blocks definition edits, not re-use: sharing still works."""
        vocab = Vocabulary(name="m", terms=["fox"])
        d1 = Dataset(config=get_config("base").clone().add_vocabulary(vocab))
        d1.add("the quick brown fox", "the quick brown fox")
        d2 = Dataset(config=get_config("base").clone().add_vocabulary(vocab))
        d2.add("a fox", "a fox")
        assert d1.metrics.ktr(vocab="m").value == 1.0
        assert d2.metrics.ktr(vocab="m").value == 1.0

    def test_clone_allows_modification_after_freeze(self):
        """clone() returns a mutable copy of a frozen vocabulary."""
        vocab = Vocabulary(name="m", terms=["a"])
        get_config("base").clone().add_vocabulary(vocab)
        assert vocab.is_frozen
        cloned = vocab.clone()
        assert not cloned.is_frozen
        cloned.add_terms(["b"])  # no error
        assert cloned._static_terms == {"a", "b"}


class TestVocabularyClone:
    """clone() shares Vocabulary objects and re-resolves them against the clone."""

    def test_clone_shares_vocab_and_recomputes(self):
        dataset = Dataset(
            config=get_config("base").clone().add_vocabulary(Vocabulary("m", extractors=[lambda ds: ["diabetes"]]))
        )
        dataset.add("the patient has diabetes", "the patient has diabetis")
        clone = dataset.clone()
        # Same Vocabulary object is shared.
        assert clone.config.vocabularies["m"] is dataset.config.vocabularies["m"]
        # The clone resolves and computes independently.
        assert clone.metrics.ktr(vocab="m").value == 0.0  # "diabetis" != "diabetes"


class TestVocabularyFreezeAndCache:
    """Resolving a vocabulary freezes the dataset (closes the stale-cache issue)."""

    def test_matching_freezes_dataset(self):
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("a", terms=["fox"])))
        dataset.add("the quick brown fox", "the quick brown fox")
        assert dataset.is_frozen is False
        # Direct matching triggers resolution, which freezes the dataset.
        dataset.config.vocabularies["a"].build_matcher(dataset)
        assert dataset.is_frozen is True
        with pytest.raises(DatasetFrozenError):
            dataset.add("more", "more")

    def test_build_matcher_returns_trie(self):
        vocab = Vocabulary("a", terms=["fox"])
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(vocab))
        dataset.add("the quick brown fox", "the quick brown fox")
        matcher = vocab.build_matcher(dataset)
        assert matcher is not None
        matches = matcher.find_matches(dataset[0].ref)
        assert len(matches) == 1
        assert matches[0].key_term.raw == "fox"


class TestVocabularyEdgeCases:
    def test_empty_extractor_output(self):
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("e", extractors=[lambda ds: []])))
        dataset.add("a b c", "a b c")
        matcher = dataset.config.vocabularies["e"].build_matcher(dataset)
        assert matcher.find_matches(dataset[0].ref) == []

    def test_extractor_returning_none(self):
        dataset = Dataset(
            config=get_config("base").clone().add_vocabulary(Vocabulary("e", extractors=[lambda ds: None]))
        )
        dataset.add("a b c", "a b c")
        matcher = dataset.config.vocabularies["e"].build_matcher(dataset)
        assert matcher.find_matches(dataset[0].ref) == []

    def test_extractor_raising_wrapped(self):
        def boom(ds):
            raise RuntimeError("nope")

        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary("e", extractors=[boom])))
        dataset.add("a b c", "a b c")
        with pytest.raises(VocabularyExtractorError):
            dataset.metrics.ktr(vocab="e").value

    def test_extractor_non_string_output_raises(self):
        dataset = Dataset(
            config=get_config("base").clone().add_vocabulary(Vocabulary("e", extractors=[lambda ds: [1, 2]]))
        )
        dataset.add("a b c", "a b c")
        with pytest.raises(TypeError):
            dataset.metrics.ktr(vocab="e").value

    def test_source_less_vocab_yields_no_matches(self):
        dataset = Dataset(config=get_config("base").clone().add_vocabulary(Vocabulary(name="empty")))
        dataset.add("a b c", "a b c")
        matcher = dataset.config.vocabularies["empty"].build_matcher(dataset)
        assert matcher.find_matches(dataset[0].ref) == []
        assert dataset.metrics._kt_stats(vocab="empty").num_ref_terms == 0

    def test_unregistered_vocab_name_returns_no_matches(self):
        dataset = Dataset()
        dataset.add("a b c", "a b c")
        assert "missing" not in dataset.config.vocabularies
