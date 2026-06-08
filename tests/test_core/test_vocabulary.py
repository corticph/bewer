import tempfile
from pathlib import Path

import pytest

from bewer.core.dataset import Dataset
from bewer.core.vocabulary import Vocabulary


def _matched_token_lists(matches, text):
    """Return the raw token lists for each match's span, sorted for stable comparison."""
    return sorted(text.tokens[m.span].raw for m in matches)


class TestVocabularyConstruction:
    def test_from_list_registers_and_matches(self):
        ds = Dataset()
        ds.add("the quick brown fox", "the quick brown dog")
        ds.add_vocabulary_from_list("animals", ["fox"])
        assert ds.has_vocabulary("animals")
        assert isinstance(ds.get_vocabulary("animals"), Vocabulary)
        assert len(ds[0].ref.get_key_term_matches("animals")) == 1

    def test_from_file_reads_one_term_per_line(self):
        ds = Dataset()
        ds.add("the quick brown fox", "the quick brown dog")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("fox\nbrown\n")
            path = f.name
        try:
            ds.add_vocabulary_from_file("animals", path)
            kt_raws = {kt.raw for kt in ds.get_vocabulary("animals").key_terms}
            assert kt_raws == {"fox", "brown"}
        finally:
            Path(path).unlink()

    def test_from_list_rejects_string(self):
        with pytest.raises(TypeError):
            Vocabulary.from_list("v", "not-a-list")

    def test_global_terms_match_every_example(self):
        """Explicit (global) terms have no example association and match everywhere by default."""
        ds = Dataset()
        ds.add("the fox", "the fox")
        ds.add("another fox here", "another fox here")
        ds.add_vocabulary_from_list("animals", ["fox"])
        assert len(ds[0].ref.get_key_term_matches("animals")) == 1
        assert len(ds[1].ref.get_key_term_matches("animals")) == 1

    def test_global_terms_have_empty_example_backrefs(self):
        ds = Dataset()
        ds.add("the fox", "the fox")
        ds.add_vocabulary_from_list("animals", ["fox"])
        (kt,) = ds.get_vocabulary("animals").key_terms
        assert kt.examples == set()


class TestPerExampleAnnotations:
    def test_annotation_terms_track_their_example(self):
        ds = Dataset()
        ds.add("paracetamol helps", "paracetamol helps", key_terms={"drugs": ["paracetamol"]})
        vocab = ds.get_vocabulary("drugs")
        (kt,) = vocab.key_terms
        assert kt.raw == "paracetamol"
        assert kt.examples == {ds[0]}

    def test_example_key_terms_is_derived_view(self):
        ds = Dataset()
        ds.add("the quick brown fox", "the quick brown fox", key_terms={"animals": ["fox"], "colors": ["brown"]})
        example = ds[0]
        assert set(example.key_terms.keys()) == {"animals", "colors"}
        assert {kt.raw for kt in example.key_terms["animals"]} == {"fox"}

    def test_canonical_dedup_across_examples(self):
        """Two examples annotating the same raw term share one canonical KeyTerm instance."""
        ds = Dataset()
        ds.add("fox one", "fox one", key_terms={"animals": ["fox"]})
        ds.add("fox two", "fox two", key_terms={"animals": ["fox"]})
        vocab = ds.get_vocabulary("animals")
        (kt,) = vocab.key_terms
        assert kt.examples == {ds[0], ds[1]}


class TestMixedSources:
    def test_explicit_and_annotation_terms_union(self):
        """A vocabulary may hold explicit (global) terms and per-example annotations together."""
        ds = Dataset()
        ds.add_vocabulary_from_list("terms", ["alpha"])
        ds.add("alpha beta", "alpha beta", key_terms={"terms": ["beta"]})
        vocab = ds.get_vocabulary("terms")
        assert {kt.raw for kt in vocab.key_terms} == {"alpha", "beta"}
        # alpha is global (no examples); beta is regarded by the example.
        by_raw = {kt.raw: kt for kt in vocab.key_terms}
        assert by_raw["alpha"].examples == set()
        assert by_raw["beta"].examples == {ds[0]}


class TestLocalOnlyMatches:
    def test_default_matches_everywhere(self):
        """By default (only_local_matches=False) every union term matches in every text."""
        ds = Dataset()
        ds.add("paracetamol helps", "paracetamol helps", key_terms={"drugs": ["paracetamol"]})
        ds.add("ibuprofen and paracetamol", "ibuprofen and paracetamol", key_terms={"drugs": ["ibuprofen"]})
        # Example 1 only annotates ibuprofen, but paracetamol (from example 0) still matches here.
        matched = _matched_token_lists(ds[1].ref.get_key_term_matches("drugs"), ds[1].ref)
        assert ["ibuprofen"] in matched
        assert ["paracetamol"] in matched

    def test_local_only_scopes_to_regarding_example(self):
        """The 'period' case: a term regarded by example 0 is dropped in example 1 under local_only."""
        ds = Dataset()
        ds.add("paracetamol helps", "paracetamol helps", key_terms={"drugs": ["paracetamol"]})
        ds.add("ibuprofen and paracetamol", "ibuprofen and paracetamol", key_terms={"drugs": ["ibuprofen"]})
        matched = _matched_token_lists(ds[1].ref.get_key_term_matches("drugs", only_local_matches=True), ds[1].ref)
        assert ["ibuprofen"] in matched
        assert ["paracetamol"] not in matched
        # ...but the term is still kept in the example that regards it.
        matched0 = _matched_token_lists(ds[0].ref.get_key_term_matches("drugs", only_local_matches=True), ds[0].ref)
        assert ["paracetamol"] in matched0

    def test_local_only_example_with_no_terms_matches_nothing(self):
        """An example regarding no terms yields nothing under local_only, even if a term appears verbatim."""
        ds = Dataset()
        ds.add("paracetamol helps", "paracetamol helps", key_terms={"drugs": ["paracetamol"]})
        ds.add("the paracetamol talk", "the paracetamol talk", key_terms={"drugs": []})
        assert ds[1].ref.get_key_term_matches("drugs", only_local_matches=True) == []

    def test_local_only_global_terms_yield_nothing(self):
        """Global-only terms have empty .examples, so only_local_matches yields nothing for them."""
        ds = Dataset()
        ds.add("the fox", "the fox")
        ds.add_vocabulary_from_list("animals", ["fox"])
        assert ds[0].ref.get_key_term_matches("animals", only_local_matches=True) == []
        assert len(ds[0].ref.get_key_term_matches("animals", only_local_matches=False)) == 1


class TestFunctionExtraction:
    def test_extractor_is_lazy_and_cached(self):
        ds = Dataset()
        ds.add("paracetamol and ibuprofen", "paracetamol and ibuprofen")
        calls = []

        def extractor(dataset):
            calls.append(1)
            return {i: ex.ref.raw.split() for i, ex in enumerate(dataset)}

        ds.add_vocabulary_from_function("derived", extractor)
        assert calls == []  # not invoked until terms are needed

        ds[0].ref.get_key_term_matches("derived")
        assert len(calls) == 1

        ds[0].ref.get_key_term_matches("derived")
        assert len(calls) == 1  # cached, not re-extracted

    def test_non_mapping_extractor_result_raises(self):
        ds = Dataset()
        ds.add("alpha beta", "alpha beta")
        ds.add_vocabulary_from_function("derived", lambda d: ["alpha"])
        with pytest.raises(TypeError):
            ds.get_vocabulary("derived").key_terms

    def test_local_extractor_mapping_associates_terms_with_examples(self):
        """A mapping of example index -> terms produces local terms scoped to those examples."""
        ds = Dataset()
        ds.add("alpha here", "alpha here")
        ds.add("beta there", "beta there")
        ds.add_vocabulary_from_function("derived", lambda d: {0: ["alpha"], 1: ["beta"]})
        vocab = ds.get_vocabulary("derived")
        by_raw = {kt.raw: kt for kt in vocab.key_terms}
        assert by_raw["alpha"].examples == {ds[0]}
        assert by_raw["beta"].examples == {ds[1]}
        # Under local_only, each example matches only its own term.
        matched = _matched_token_lists(ds[0].ref.get_key_term_matches("derived", only_local_matches=True), ds[0].ref)
        assert matched == [["alpha"]]

    def test_extractor_reruns_after_mutation(self):
        ds = Dataset()
        ds.add("alpha", "alpha")
        extractions = []

        def extractor(dataset):
            terms = {i: [ex.ref.raw] for i, ex in enumerate(dataset)}
            extractions.append({raw for raws in terms.values() for raw in raws})
            return terms

        ds.add_vocabulary_from_function("derived", extractor)
        assert len(ds[0].ref.get_key_term_matches("derived")) == 1

        ds.add("beta", "beta")  # mutation invalidates the cached extraction
        assert len(ds[1].ref.get_key_term_matches("derived")) == 1
        assert extractions == [{"alpha"}, {"alpha", "beta"}]

    def test_empty_extractor_result_is_noop(self):
        ds = Dataset()
        ds.add("alpha", "alpha")
        ds.add_vocabulary_from_function("derived", lambda d: {})
        assert ds.get_vocabulary("derived").key_terms == set()
        assert ds[0].ref.get_key_term_matches("derived") == []


class TestMatchKeyTerms:
    def test_collision_returns_multiple_key_terms(self):
        """A span maps to multiple key terms when distinct raw strings normalize identically."""
        ds = Dataset()
        ds.add("the quick brown fox", "the quick brown fox")
        ds.add_vocabulary_from_list("animals", ["fox", "FOX"])
        matches = ds[0].ref.get_key_term_matches("animals")
        assert len(matches) == 1
        assert {kt.raw for kt in matches[0].key_terms} == {"fox", "FOX"}


class TestVocabularyRegistration:
    def test_duplicate_name_folds_into_existing_instance(self):
        """Registering an existing name folds in its terms and returns the same instance."""
        ds = Dataset()
        ds.add("a b", "a b")
        first = ds.add_vocabulary_from_list("v", ["a"])
        second = ds.add_vocabulary_from_list("v", ["b"])
        assert second is first
        assert {kt.raw for kt in ds.get_vocabulary("v").key_terms} == {"a", "b"}

    def test_register_over_annotation_name_combines(self):
        """Registering a vocabulary over an annotation-registered name folds in its terms."""
        ds = Dataset()
        ds.add("the fox", "the fox", key_terms={"animals": ["fox"]})
        annotation_vocab = ds.get_vocabulary("animals")
        registered = ds.add_vocabulary_from_list("animals", ["dog"])
        assert registered is annotation_vocab
        assert {kt.raw for kt in ds.get_vocabulary("animals").key_terms} == {"fox", "dog"}

    def test_annotation_after_explicit_merges(self):
        """Annotating an existing vocabulary's name reuses it and merges the terms."""
        ds = Dataset()
        ds.add_vocabulary_from_list("animals", ["dog"])
        ds.add("the fox", "the fox", key_terms={"animals": ["fox"]})
        assert {kt.raw for kt in ds.get_vocabulary("animals").key_terms} == {"dog", "fox"}

    def test_get_vocabulary_unknown_raises(self):
        ds = Dataset()
        with pytest.raises(ValueError, match="not found in dataset key term vocabularies"):
            ds.get_vocabulary("nope")

    def test_add_vocabulary_returns_bound_vocabulary(self):
        ds = Dataset()
        vocab = ds.add_vocabulary(Vocabulary.from_list("animals", ["fox"]))
        assert vocab.dataset is ds


class TestVocabularyCombine:
    def test_list_and_function_sources_combine(self):
        ds = Dataset()
        ds.add("alpha beta", "alpha beta")
        ds.add_vocabulary_from_list("v", ["alpha"])
        ds.add_vocabulary_from_function("v", lambda d: {0: ["beta"]})
        assert {kt.raw for kt in ds.get_vocabulary("v").key_terms} == {"alpha", "beta"}

    def test_file_and_function_sources_combine(self):
        ds = Dataset()
        ds.add("alpha beta", "alpha beta")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("alpha\n")
            path = f.name
        try:
            ds.add_vocabulary_from_file("v", path)
            ds.add_vocabulary_from_function("v", lambda d: {0: ["beta"]})
            assert {kt.raw for kt in ds.get_vocabulary("v").key_terms} == {"alpha", "beta"}
        finally:
            Path(path).unlink()

    def test_two_extractors_both_run_and_union(self):
        ds = Dataset()
        ds.add("alpha beta", "alpha beta")
        ds.add_vocabulary_from_function("v", lambda d: {0: ["alpha"]})
        ds.add_vocabulary_from_function("v", lambda d: {0: ["beta"]})
        assert {kt.raw for kt in ds.get_vocabulary("v").key_terms} == {"alpha", "beta"}

    def test_combine_returns_detached_union(self):
        a = Vocabulary.from_list("v", ["alpha"])
        b = Vocabulary.from_function("v", lambda d: {0: ["beta"]})
        combined = a.combine(b)
        assert combined.name == "v"
        assert combined.dataset is None  # detached until registered
        ds = Dataset()
        ds.add("alpha beta", "alpha beta")
        ds.add_vocabulary(combined)
        assert {kt.raw for kt in ds.get_vocabulary("v").key_terms} == {"alpha", "beta"}

    def test_add_operator_combines(self):
        ds = Dataset()
        ds.add("alpha beta", "alpha beta")
        ds.add_vocabulary(Vocabulary.from_list("v", ["alpha"]) + Vocabulary.from_list("v", ["beta"]))
        assert {kt.raw for kt in ds.get_vocabulary("v").key_terms} == {"alpha", "beta"}

    def test_combine_across_names_with_explicit_name(self):
        ds = Dataset()
        ds.add("alpha beta", "alpha beta")
        drugs = Vocabulary.from_list("drugs", ["alpha"])
        anatomy = Vocabulary.from_list("anatomy", ["beta"])
        ds.add_vocabulary(drugs.combine(anatomy, name="medical"))
        assert {kt.raw for kt in ds.get_vocabulary("medical").key_terms} == {"alpha", "beta"}

    def test_shared_term_reports_multiple_vocabularies(self):
        """A term reachable via several registered vocabularies is one canonical object."""
        ds = Dataset()
        ds.add("alpha beta", "alpha beta")
        ds.add_vocabulary_from_list("drugs", ["alpha"])
        ds.add_vocabulary_from_list("anatomy", ["beta"])
        ds.add_vocabulary(ds.get_vocabulary("drugs").combine(ds.get_vocabulary("anatomy"), name="medical"))
        (alpha,) = (kt for kt in ds.key_terms.values() if kt.raw == "alpha")
        assert alpha is ds.key_terms["alpha"]
        assert {v.name for v in alpha.vocabularies} == {"drugs", "medical"}
        assert alpha in ds.get_vocabulary("drugs").key_terms
        assert alpha in ds.get_vocabulary("medical").key_terms

    def test_combine_preserves_annotations(self):
        """Combining vocabularies that carry per-example annotations does not lose them."""
        ds = Dataset()
        ds.add("the alpha", "the alpha", key_terms={"drugs": ["alpha"]})
        ds.add("the beta", "the beta", key_terms={"anatomy": ["beta"]})
        ds.add_vocabulary(ds.get_vocabulary("drugs").combine(ds.get_vocabulary("anatomy"), name="medical"))
        medical = ds.get_vocabulary("medical")
        assert {kt.raw for kt in medical.key_terms} == {"alpha", "beta"}
        # only_local scoping still works per example through the combined vocabulary.
        assert len(ds[0].ref.get_key_term_matches("medical", only_local_matches=True)) == 1
        assert len(ds[1].ref.get_key_term_matches("medical", only_local_matches=True)) == 1

    def test_combine_result_is_detached_even_when_operands_bound(self):
        ds = Dataset()
        ds.add("alpha beta", "alpha beta")
        a = ds.add_vocabulary_from_list("v", ["alpha"])
        assert a.dataset is ds
        combined = a.combine(Vocabulary.from_list("v", ["beta"]))
        assert combined.dataset is None


class TestCacheInvalidation:
    def test_adding_example_invalidates_cached_metric(self):
        ds = Dataset()
        ds.add("the fox", "the dog", key_terms={"animals": ["fox"]})
        ktr_before = ds.metrics.ktr(vocab="animals").value
        assert ktr_before == 0.0  # the single key term was missed

        ds.add("the fox", "the fox", key_terms={"animals": ["fox"]})
        ktr_after = ds.metrics.ktr(vocab="animals").value
        assert ktr_after == pytest.approx(0.5)  # recomputed over both examples

    def test_adding_vocabulary_invalidates_cached_matches(self):
        ds = Dataset()
        ds.add("the quick brown fox", "the quick brown fox")
        ds.add_vocabulary_from_list("a", ["fox"])
        assert len(ds[0].ref.get_key_term_matches("a")) == 1
        # Registering another vocabulary invalidates caches; existing lookups still work.
        ds.add_vocabulary_from_list("b", ["brown"])
        assert len(ds[0].ref.get_key_term_matches("b")) == 1
        assert len(ds[0].ref.get_key_term_matches("a")) == 1
