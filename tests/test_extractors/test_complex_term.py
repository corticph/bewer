"""Tests for bewer.extractors.complex_term: the COMPLEX_TERM_DEFAULT_PATTERN, the
match_token_regex primitive, and the ComplexTermExtractor."""

import pytest
import regex as re

from bewer import Dataset
from bewer.extractors import ComplexTermExtractor
from bewer.extractors.complex_term import COMPLEX_TERM_DEFAULT_PATTERN, match_token_regex
from bewer.preprocessing.context import set_pipeline

DEFAULT = re.compile(COMPLEX_TERM_DEFAULT_PATTERN)


def ct_tokens(dataset, index=0):
    """ref tokens under the complex_term tokenizer (no hyphen split), as the CT metrics use.

    A hyphenated compound such as ``CT-scan`` is therefore a single token here.
    """
    with set_pipeline(tokenizer="complex_term"):
        return dataset[index].ref.tokens


def extract(extractor, dataset):
    """Run an extractor under the complex_term tokenizer, as the CT metrics do."""
    with set_pipeline(tokenizer="complex_term"):
        return extractor(dataset)


class TestComplexTermDefaultPattern:
    """COMPLEX_TERM_DEFAULT_PATTERN matches case-distinctive / alphanumeric / Greek tokens
    (including hyphenated compounds, which the complex_term tokenizer keeps whole) and rejects
    ordinary words. These exercise the regex directly via ``fullmatch`` on the string."""

    @pytest.mark.parametrize(
        "token",
        [
            "MRI",  # all-caps acronyms (>=2 uppercase in a segment)
            "NATO",
            "mmHg",  # mixed-case, lowercase-first
            "iPhone",
            "mRNA",
            "HbA1c",  # mixed-case, uppercase-first
            "DNase",
            "CO2",  # letter-then-digit
            "B12",
            "3D",  # digit-then-uppercase
            "5HT",
            "b12",  # all-lowercase letter+digit (so ref/hyp case mismatches surface)
            "hello1",
            "THE",  # all-caps everyday words (documented false positives)
            "OK",
            "μM",  # Greek anywhere
            "ΔG",
            "γδ",
            "CT-scan",  # hyphenated: >=2 uppercase in a segment
            "pre-MRI",
            "X-ray",  # hyphenated: lone uppercase-letter segment
            "D-glucose",
            "T-shirt",
            "5-HT-receptor",  # hyphenated: digit promotes whole compound
            "random-b12-text",
            "α-helix",  # hyphenated: Greek anywhere
        ],
    )
    def test_matches(self, token):
        assert DEFAULT.fullmatch(token) is not None, f"expected match for {token!r}"

    @pytest.mark.parametrize(
        "token",
        [
            "Patient",  # ordinary capitalised / lowercase words
            "hello",
            "A",  # single letters
            "I",
            "1st",  # ordinals (en / fr)
            "1er",
            "2024",  # pure digits
            "Hello-World",  # title-case compounds (one uppercase per segment)
            "Patient-Care",
            "up-to-date",  # lowercase compounds
            "e-mail",
        ],
    )
    def test_rejects(self, token):
        assert DEFAULT.fullmatch(token) is None, f"expected reject for {token!r}"


class TestMatchTokenRegex:
    """match_token_regex full-matches each token's case-preserving ``raw`` and returns unit
    slices. Under the complex_term tokenizer a hyphenated compound is one token, so there is
    no cross-token grouping or sub-token fallback."""

    def test_single_and_multiple_matches(self):
        dataset = Dataset()
        dataset.add(ref="MRI shows HbA1c is 7.2 mmHg CH3", hyp="")
        tokens = ct_tokens(dataset)
        matches = match_token_regex(tokens, DEFAULT)
        assert [tokens[m.start].raw for m in matches] == ["MRI", "HbA1c", "mmHg", "CH3"]
        assert all(m.stop - m.start == 1 for m in matches)

    def test_no_matches(self):
        dataset = Dataset()
        dataset.add(ref="the patient had a routine exam", hyp="")
        assert match_token_regex(ct_tokens(dataset), DEFAULT) == []

    def test_uses_raw_case_sensitively(self):
        """Matching uses Token.raw, so a lowercased acronym does not match the cased rule."""
        upper = Dataset()
        upper.add(ref="had an MRI", hyp="")
        lower = Dataset()
        lower.add(ref="had an mri", hyp="")
        assert len(match_token_regex(ct_tokens(upper), DEFAULT)) == 1
        assert len(match_token_regex(ct_tokens(lower), DEFAULT)) == 0

    def test_fullmatch_not_substring(self):
        dataset = Dataset()
        dataset.add(ref="preMRI exam", hyp="")
        # "preMRI" contains "MRI" but is not a fullmatch for it.
        assert match_token_regex(ct_tokens(dataset), re.compile(r"MRI")) == []

    def test_custom_pattern(self):
        dataset = Dataset()
        dataset.add(ref="alpha bravo charlie", hyp="")
        assert match_token_regex(ct_tokens(dataset), re.compile(r"alpha")) == [slice(0, 1)]

    def test_empty(self):
        dataset = Dataset()
        dataset.add(ref="", hyp="")
        assert match_token_regex(ct_tokens(dataset), DEFAULT) == []


class TestHyphenatedCompoundsAreSingleTokens:
    """Under the complex_term tokenizer a hyphenated compound is one token, matched whole."""

    @pytest.mark.parametrize(
        "ref, term",
        [
            ("patient had a CT-scan", "CT-scan"),
            ("patient had an X-ray", "X-ray"),
            ("the pre-MRI screening", "pre-MRI"),
            ("the 5-HT-receptor pathway", "5-HT-receptor"),
            ("wearing a T-shirt today", "T-shirt"),
            ("here is some random-b12-text", "random-b12-text"),
        ],
    )
    def test_compound_is_single_token_match(self, ref, term):
        dataset = Dataset()
        dataset.add(ref=ref, hyp="")
        tokens = ct_tokens(dataset)
        matches = match_token_regex(tokens, DEFAULT)
        assert len(matches) == 1
        assert tokens[matches[0].start].raw == term

    def test_multiple_compounds_and_acronym(self):
        dataset = Dataset()
        dataset.add(ref="MRI shows a CT-scan and an X-ray", hyp="")
        tokens = ct_tokens(dataset)
        matched = [tokens[m.start].raw for m in match_token_regex(tokens, DEFAULT)]
        assert matched == ["MRI", "CT-scan", "X-ray"]

    @pytest.mark.parametrize(
        "ref",
        [
            "the Hello-World example",  # title-case compound
            "keep it up-to-date please",  # lowercase compound
            "send me an e-mail",
            "patient had an X - ray",  # spaced: tokenizes to separate X, ray — neither matches
        ],
    )
    def test_no_match(self, ref):
        dataset = Dataset()
        dataset.add(ref=ref, hyp="")
        assert match_token_regex(ct_tokens(dataset), DEFAULT) == []

    def test_no_subtoken_fallback(self):
        """A pattern matching only part of a compound token does not register; the whole token
        must fullmatch."""
        dataset = Dataset()
        dataset.add(ref="alpha-bravo-charlie", hyp="")
        tokens = ct_tokens(dataset)
        assert match_token_regex(tokens, re.compile(r"bravo")) == []
        assert match_token_regex(tokens, re.compile(r"alpha-bravo-charlie")) == [slice(0, 1)]


class TestComplexTermExtractor:
    """ComplexTermExtractor as a VocabularyExtractor (run under the complex_term tokenizer)."""

    def test_extracts_per_example_preserving_case(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI and HbA1c", hyp="")
        dataset.add(ref="the patient rested well", hyp="")
        # Case preserved (HbA1c, not hba1c); example 1 has no complex terms (omitted).
        assert extract(ComplexTermExtractor(), dataset) == {0: {"MRI", "HbA1c"}}

    def test_hyphen_compound_kept_as_one_term(self):
        dataset = Dataset()
        dataset.add(ref="ordered a CT-scan today", hyp="")
        assert extract(ComplexTermExtractor(), dataset)[0] == {"CT-scan"}

    def test_empty_when_no_complex_terms(self):
        dataset = Dataset()
        dataset.add(ref="the quick brown fox", hyp="")
        assert extract(ComplexTermExtractor(), dataset) == {}

    def test_custom_pattern(self):
        dataset = Dataset()
        dataset.add(ref="alpha BRAVO charlie", hyp="")
        assert extract(ComplexTermExtractor(re.compile(r"alpha")), dataset) == {0: {"alpha"}}

    def test_registers_as_function_vocabulary(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI", hyp="patient had an MRI")
        vocab = dataset.add_vocabulary_from_function("complex", ComplexTermExtractor())
        assert "MRI" in {kt.raw for kt in vocab.key_terms}

    def test_repr(self):
        assert "ComplexTermExtractor" in repr(ComplexTermExtractor())
