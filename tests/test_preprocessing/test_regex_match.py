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
            # Single-letter tokens
            "A",
            "I",
            "μ",
            "α",
            "β",
            # All-lowercase mixed-script (no case signal, no digit)
            "μg",
            "αv",
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

    @pytest.mark.parametrize("token", ["THE", "STOP", "NO"])
    def test_documented_false_positives(self, token):
        """Shouted ordinary words match the all-uppercase branch — accepted limitation."""
        assert DEFAULT.fullmatch(token) is not None

    @pytest.mark.parametrize("token", ["β2", "o2", "b12", "hello1"])
    def test_all_lowercase_letter_digit_match(self, token):
        """All-lowercase letter+digit tokens are caught by the second branch so case mismatches
        between ref and hyp register on the alignment."""
        assert DEFAULT.fullmatch(token) is not None


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
