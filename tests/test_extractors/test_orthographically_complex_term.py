"""Tests for bewer.extractors.orthographically_complex_term and the generic regex extractor."""

import pytest
import regex

from bewer import Dataset, Vocabulary
from bewer.extractors import (
    ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN,
    OrthographicallyComplexTermExtractor,
    RegexExtractor,
    match_token_regex,
)
from bewer.preprocessing.context import set_pipeline


@pytest.fixture
def complex_term_context():
    """Activate the pipeline the complex-term metrics run under (no hyphen split, case kept)."""
    with set_pipeline(tokenizer="complex_term", normalizer="cased"):
        yield


def _ref_tokens(dataset):
    """Reference tokens of the first example, under the active pipeline."""
    return dataset[0].ref.tokens


class TestComplexTermDefaultPattern:
    """The default pattern accepts complex terms and rejects ordinary words/numbers."""

    @pytest.fixture(scope="class")
    def pattern(self):
        return regex.compile(ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN)

    @pytest.mark.parametrize(
        "term",
        [
            "MRI",
            "NATO",
            "mmHg",
            "iPhone",
            "mRNA",
            "HbA1c",
            "DNase",
            "CO2",
            "B12",
            "3D",
            "5HT",
            "b12",
            "hello1",
            "THE",
            "OK",
            "μM",
            "ΔG",
            "γδ",
            "CT-scan",
            "pre-MRI",
            "X-ray",
            "D-glucose",
            "T-shirt",
            "5-HT-receptor",
            "random-b12-text",
            "α-helix",
        ],
    )
    def test_accepts_complex_terms(self, pattern, term):
        assert pattern.fullmatch(term) is not None

    @pytest.mark.parametrize(
        "term",
        ["Patient", "hello", "A", "I", "1st", "1er", "2024", "Hello-World", "Patient-Care", "up-to-date", "e-mail"],
    )
    def test_rejects_ordinary_tokens(self, pattern, term):
        assert pattern.fullmatch(term) is None


class TestMatchTokenRegex:
    """match_token_regex returns unit slices for fully-matching tokens."""

    def test_returns_slices_for_matches(self, complex_term_context):
        dataset = Dataset()
        dataset.add("the MRI showed CO2", "x")
        tokens = _ref_tokens(dataset)
        pattern = regex.compile(ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN)
        spans = match_token_regex(tokens, pattern)
        matched = [tokens[s.start].raw for s in spans]
        assert all(s.stop - s.start == 1 for s in spans)
        assert matched == ["MRI", "CO2"]

    def test_no_matches_returns_empty(self, complex_term_context):
        dataset = Dataset()
        dataset.add("the patient is here", "x")
        tokens = _ref_tokens(dataset)
        assert match_token_regex(tokens, regex.compile(ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN)) == []

    def test_is_full_match_not_substring(self, complex_term_context):
        dataset = Dataset()
        dataset.add("preMRItext", "x")  # contains MRI but token as a whole is matched
        tokens = _ref_tokens(dataset)
        # "preMRItext" has lowercase-then-uppercase evidence, so it matches as a whole token.
        spans = match_token_regex(tokens, regex.compile(ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN))
        assert [tokens[s.start].raw for s in spans] == ["preMRItext"]

    def test_custom_pattern(self, complex_term_context):
        dataset = Dataset()
        dataset.add("alpha BETA gamma", "x")
        tokens = _ref_tokens(dataset)
        spans = match_token_regex(tokens, regex.compile(r"\p{Lu}+"))  # all-uppercase tokens
        assert [tokens[s.start].raw for s in spans] == ["BETA"]


class TestRegexExtractorBase:
    """The generic RegexExtractor is reusable for other regex-based term families."""

    def test_default_pattern_matches_nothing(self, complex_term_context):
        dataset = Dataset()
        dataset.add("MRI and CO2", "x")
        assert RegexExtractor()(dataset) == set()

    def test_explicit_pattern(self, complex_term_context):
        dataset = Dataset()
        dataset.add("alpha BETA GAMMA delta", "x")
        extractor = RegexExtractor(r"\p{Lu}+")
        assert extractor(dataset) == {"BETA", "GAMMA"}

    def test_subclass_overrides_default_pattern(self, complex_term_context):
        class AllCapsExtractor(RegexExtractor):
            default_pattern = r"\p{Lu}+"

        dataset = Dataset()
        dataset.add("the FDA approved it", "x")
        assert AllCapsExtractor()(dataset) == {"FDA"}

    def test_accepts_compiled_pattern(self):
        compiled = regex.compile(r"\p{Lu}+")
        assert RegexExtractor(compiled).pattern is compiled

    def test_repr_includes_pattern(self):
        assert repr(RegexExtractor("[ab]+")) == "RegexExtractor('[ab]+')"


class TestOrthographicallyComplexTermExtractor:
    """OrthographicallyComplexTermExtractor harvests complex terms from references as a global term set."""

    def test_extracts_complex_terms(self, complex_term_context):
        dataset = Dataset()
        dataset.add("the patient had an MRI and a CO2 test", "x")
        assert OrthographicallyComplexTermExtractor()(dataset) == {"MRI", "CO2"}

    def test_preserves_case(self, complex_term_context):
        dataset = Dataset()
        dataset.add("measured HbA1c and mmHg", "x")
        assert OrthographicallyComplexTermExtractor()(dataset) == {"HbA1c", "mmHg"}

    def test_keeps_hyphen_compounds_whole(self, complex_term_context):
        dataset = Dataset()
        dataset.add("ordered a CT-scan and an X-ray", "x")
        assert OrthographicallyComplexTermExtractor()(dataset) == {"CT-scan", "X-ray"}

    def test_unions_terms_across_examples(self, complex_term_context):
        dataset = Dataset()
        dataset.add("an MRI scan", "x")
        dataset.add("a CO2 reading", "x")
        assert OrthographicallyComplexTermExtractor()(dataset) == {"MRI", "CO2"}

    def test_empty_when_no_complex_terms(self, complex_term_context):
        dataset = Dataset()
        dataset.add("the patient is doing well", "x")
        assert OrthographicallyComplexTermExtractor()(dataset) == set()

    def test_custom_pattern(self, complex_term_context):
        dataset = Dataset()
        dataset.add("MRI and CO2", "x")
        extractor = OrthographicallyComplexTermExtractor(pattern=r"\p{Lu}+")  # acronyms only, no alphanumerics
        assert extractor(dataset) == {"MRI"}

    def test_registers_as_function_vocabulary(self):
        dataset = Dataset()
        dataset.add("the MRI and CT-scan", "the MRI and CT-scan")
        dataset.add_vocabulary(Vocabulary("cts").add_extractor(OrthographicallyComplexTermExtractor()))
        matches = dataset[0].ref.get_key_term_matches(vocab="cts", normalized=True)
        # Resolved under the default key-term-ish context; just assert the vocab resolves.
        assert isinstance(matches, list)

    def test_repr(self):
        assert (
            repr(OrthographicallyComplexTermExtractor())
            == f"OrthographicallyComplexTermExtractor({ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN!r})"
        )
