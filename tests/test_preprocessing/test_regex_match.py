"""Tests for bewer.preprocessing.regex_match module."""

import pytest
import regex as re

from bewer import Dataset
from bewer.preprocessing.regex_match import ALPHANUM_DEFAULT_PATTERN, match_token_regex

DEFAULT = re.compile(ALPHANUM_DEFAULT_PATTERN)


class TestAlphaNumDefaultPattern:
    """Tests that the default ALPHANUM_DEFAULT_PATTERN matches and rejects expected token forms."""

    @pytest.mark.parametrize(
        "token",
        [
            # Initialisms / acronyms (all-caps)
            "MRI",
            "ECG",
            "NATO",
            "FBI",
            "USA",
            "IBM",
            # Mixed-case (lowercase first)
            "mmHg",
            "iPhone",
            "mRNA",
            "eGFR",
            "iPad",
            "mAb",
            # Mixed-case (uppercase first with internal uppercase or digit)
            "HbS",
            "HbA1c",
            "IgG",
            "DNase",
            # Letter-then-digit
            "CH3",
            "B12",
            "CO2",
            "O2",
            "T4",
            "R2",
            # Digit-then-letter (with uppercase)
            "3D",
            "5G",
            "5HT",
            "2D",
            # Greek-integrated
            "μM",
            "ΔG",
            "ΔΩ",
            "α2A",
            # All-uppercase ordinary words — also case-distinctive by definition
            "THE",
            "STOP",
            "NO",
        ],
    )
    def test_matches(self, token):
        assert DEFAULT.fullmatch(token) is not None, f"expected match for {token!r}"

    @pytest.mark.parametrize(
        "token",
        [
            # Ordinary capitalised English words — must NOT match
            "Patient",
            "Hello",
            "The",
            "Apple",
            "World",
            # All-lowercase ordinary words
            "hello",
            "world",
            "patient",
            # Single Latin letters
            "A",
            "I",
            # Ordinals (English + French + Dutch)
            "1st",
            "2nd",
            "3rd",
            "4th",
            "5th",
            "1er",
            "1e",
            # Pure digits
            "2024",
            "100",
            "5",
        ],
    )
    def test_rejects(self, token):
        assert DEFAULT.fullmatch(token) is None, f"expected reject for {token!r}"

    @pytest.mark.parametrize(
        "token",
        [
            # Single Greek letters (any Greek character makes a token an entity)
            "α",
            "β",
            "γ",
            "δ",
            "μ",
            "Δ",
            "Ω",
            # Greek + Latin lowercase
            "μg",
            "αv",
            # All Greek
            "γδ",
        ],
    )
    def test_greek_tokens_match(self, token):
        """Any token containing a Greek letter is treated as an entity."""
        assert DEFAULT.fullmatch(token) is not None, f"expected match for {token!r}"

    @pytest.mark.parametrize("token", ["β2", "o2", "b12", "hello1"])
    def test_all_lowercase_letter_digit_match(self, token):
        """All-lowercase letter+digit tokens are caught by the second branch so case mismatches
        between ref and hyp register on the alignment."""
        assert DEFAULT.fullmatch(token) is not None


class TestHyphenatedCompoundsAtRegexLevel:
    """Tests that the default pattern handles hyphen-joined compound strings.

    These check the regex itself (operating on the joined compound string), not the
    multi-token slicing — TestMatchTokenRegex covers the helper end-to-end.
    """

    @pytest.mark.parametrize(
        "compound",
        [
            # Abbreviation + word
            "CT-scan",
            "X-ray",
            "T-cell",
            "B-cell",
            "D-glucose",
            "L-glucose",
            # Prefix + abbreviation
            "pre-MRI",
            "non-COVID",
            "post-MI",
            # Multi-part with digits
            "5-HT",
            "vitamin-D",
            "pre-COVID-19",
            # Abbreviation + abbreviation
            "MRI-CT",
            # Mixed shapes
            "Hello-MRI",
            "pre-MRI-scan",
            "5-HT-receptor",
            # Single-uppercase prefix + lowercase part — structurally identical to X-ray.
            # The regex correctly identifies these as case-distinctive.
            "T-shirt",
            "D-day",
            "A-frame",
            "S-curve",
            # Greek-letter compounds (any Greek char anywhere in compound)
            "α-helix",
            "β-blocker",
            "γ-radiation",
        ],
    )
    def test_compound_matches(self, compound):
        assert DEFAULT.fullmatch(compound) is not None, f"expected match for {compound!r}"

    @pytest.mark.parametrize(
        "compound",
        [
            # Ordinary capitalised compounds
            "Hello-World",
            "Patient-Care",
            # All-lowercase compounds
            "up-to-date",
            "state-of-the-art",
            "mother-in-law",
            "cul-de-sac",
            "and-or",
            "blue-green",
            "e-mail",
            "e-commerce",
        ],
    )
    def test_compound_rejects(self, compound):
        assert DEFAULT.fullmatch(compound) is None, f"expected reject for {compound!r}"


class TestMatchTokenRegex:
    """Tests for match_token_regex against a real TokenList."""

    def test_returns_single_token_slices(self):
        dataset = Dataset()
        dataset.add(ref="patient had an MRI yesterday", hyp="patient had an MRI yesterday")
        tokens = dataset[0].ref.tokens
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == [slice(3, 4)]  # only "MRI" at index 3
        assert tokens[3].raw == "MRI"

    def test_multiple_matches(self):
        dataset = Dataset()
        dataset.add(ref="MRI shows HbA1c is 7.2 mmHg CH3", hyp="")
        tokens = dataset[0].ref.tokens
        matches = match_token_regex(tokens, DEFAULT)
        matched_words = [tokens[m.start].raw for m in matches]
        assert matched_words == ["MRI", "HbA1c", "mmHg", "CH3"]

    def test_no_matches(self):
        dataset = Dataset()
        dataset.add(ref="the patient had a routine exam", hyp="")
        tokens = dataset[0].ref.tokens
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == []

    def test_uses_raw_not_normalized(self):
        """Matching uses Token.raw (case-preserving), so a lowercased copy does NOT match
        what the case-distinctive uppercase form would match."""
        dataset_upper = Dataset()
        dataset_upper.add(ref="patient had an MRI", hyp="")
        dataset_lower = Dataset()
        dataset_lower.add(ref="patient had an mri", hyp="")
        upper_matches = match_token_regex(dataset_upper[0].ref.tokens, DEFAULT)
        lower_matches = match_token_regex(dataset_lower[0].ref.tokens, DEFAULT)
        assert len(upper_matches) == 1
        assert len(lower_matches) == 0

    def test_custom_pattern(self):
        """Caller can pass an arbitrary compiled pattern."""
        dataset = Dataset()
        dataset.add(ref="alpha bravo charlie", hyp="")
        only_alpha = re.compile(r"alpha")
        matches = match_token_regex(dataset[0].ref.tokens, only_alpha)
        assert matches == [slice(0, 1)]

    def test_fullmatch_semantics(self):
        """The helper uses fullmatch, so a substring match does not register."""
        dataset = Dataset()
        dataset.add(ref="preMRI exam", hyp="")
        tokens = dataset[0].ref.tokens
        # Token "preMRI" contains "MRI" as a substring but the WHOLE token must match the
        # default pattern. Since "preMRI" has lowercase-then-uppercase, branch 1 catches it.
        # Use a stricter pattern that only matches the literal "MRI".
        strict = re.compile(r"MRI")
        matches = match_token_regex(tokens, strict)
        assert matches == []  # "preMRI" is not a fullmatch for "MRI"


class TestHyphenatedCompoundsAtHelperLevel:
    """Tests that match_token_regex correctly groups hyphen-connected tokens into
    multi-token slices when the joined compound matches the pattern."""

    def test_ct_scan_is_two_token_slice(self):
        dataset = Dataset()
        dataset.add(ref="patient had a CT-scan", hyp="")
        tokens = dataset[0].ref.tokens
        # Tokens: patient, had, a, CT, scan
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == [slice(3, 5)]
        assert [tokens[k].raw for k in range(3, 5)] == ["CT", "scan"]

    def test_x_ray_is_two_token_slice(self):
        dataset = Dataset()
        dataset.add(ref="patient had an X-ray", hyp="")
        tokens = dataset[0].ref.tokens
        # Tokens: patient, had, an, X, ray
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == [slice(3, 5)]
        assert [tokens[k].raw for k in range(3, 5)] == ["X", "ray"]

    def test_pre_mri_is_two_token_slice(self):
        dataset = Dataset()
        dataset.add(ref="the pre-MRI screening", hyp="")
        tokens = dataset[0].ref.tokens
        matches = match_token_regex(tokens, DEFAULT)
        # Tokens: the, pre, MRI, screening
        assert matches == [slice(1, 3)]
        assert [tokens[k].raw for k in range(1, 3)] == ["pre", "MRI"]

    def test_three_part_compound(self):
        dataset = Dataset()
        dataset.add(ref="the 5-HT-receptor pathway", hyp="")
        tokens = dataset[0].ref.tokens
        # Tokens: the, 5, HT, receptor, pathway
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == [slice(1, 4)]
        assert [tokens[k].raw for k in range(1, 4)] == ["5", "HT", "receptor"]

    def test_mixed_ct_scan_and_x_ray(self):
        dataset = Dataset()
        dataset.add(ref="patient had a CT-scan and an X-ray", hyp="")
        tokens = dataset[0].ref.tokens
        matches = match_token_regex(tokens, DEFAULT)
        # Tokens: patient, had, a, CT, scan, and, an, X, ray
        assert matches == [slice(3, 5), slice(7, 9)]

    def test_compound_alongside_single_token_match(self):
        dataset = Dataset()
        dataset.add(ref="MRI shows a CT-scan", hyp="")
        tokens = dataset[0].ref.tokens
        # Tokens: MRI, shows, a, CT, scan
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == [slice(0, 1), slice(3, 5)]

    def test_ordinary_capitalised_compound_no_match(self):
        dataset = Dataset()
        dataset.add(ref="the Hello-World example", hyp="")
        tokens = dataset[0].ref.tokens
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == []

    def test_up_to_date_no_match(self):
        dataset = Dataset()
        dataset.add(ref="keep it up-to-date please", hyp="")
        tokens = dataset[0].ref.tokens
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == []

    def test_email_no_match(self):
        dataset = Dataset()
        dataset.add(ref="send me an e-mail", hyp="")
        tokens = dataset[0].ref.tokens
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == []

    def test_spaces_around_hyphen_disable_compound(self):
        """X - ray with surrounding spaces is not a compound (the gap contains spaces, not just
        hyphens). Falls back to single-token matching, which neither X (length 1) nor ray match."""
        dataset = Dataset()
        dataset.add(ref="patient had an X - ray", hyp="")
        tokens = dataset[0].ref.tokens
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == []

    def test_single_uppercase_letter_plus_lowercase_compound_matches(self):
        """T-shirt has the same case-distinctive shape as X-ray (single uppercase letter
        joined by hyphen to lowercase) and is correctly matched as a multi-token entity."""
        dataset = Dataset()
        dataset.add(ref="wearing a T-shirt today", hyp="")
        tokens = dataset[0].ref.tokens
        # Tokens: wearing, a, T, shirt, today
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == [slice(2, 4)]

    def test_compound_partial_fallback_to_single_token(self):
        """When the compound does NOT match but a single token within it does (via branch 2,
        e.g. letter+digit), the helper falls back to per-token matching within the group."""
        dataset = Dataset()
        dataset.add(ref="random-b12-text", hyp="")
        tokens = dataset[0].ref.tokens
        # Tokens: random, b12, text. Compound 'random-b12-text' has no uppercase → branch 1
        # fails. Branch 2 doesn't allow '-' in body, so compound match fails. Per-token:
        # b12 matches branch 2.
        matches = match_token_regex(tokens, DEFAULT)
        assert matches == [slice(1, 2)]
        assert tokens[1].raw == "b12"

    def test_empty_token_list(self):
        dataset = Dataset()
        dataset.add(ref="", hyp="")
        tokens = dataset[0].ref.tokens
        assert match_token_regex(tokens, DEFAULT) == []
