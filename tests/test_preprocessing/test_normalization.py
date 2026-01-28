"""Tests for bewer.preprocessing.normalization module."""

from bewer.preprocessing.normalization import (
    Normalizer,
    lowercase,
    nfc,
    remove_symbols,
    strip_punctuation,
    transliterate_latin_letters,
    transliterate_symbols,
)


class TestLowercase:
    """Tests for the lowercase() function."""

    def test_basic_lowercase(self):
        """Test basic uppercase to lowercase conversion."""
        assert lowercase("HELLO") == "hello"
        assert lowercase("Hello World") == "hello world"

    def test_already_lowercase(self):
        """Test text that is already lowercase."""
        assert lowercase("hello") == "hello"

    def test_empty_string(self):
        """Test empty string input."""
        assert lowercase("") == ""

    def test_mixed_case_with_numbers(self):
        """Test mixed case with numbers (numbers unchanged)."""
        assert lowercase("Hello123World") == "hello123world"

    def test_unicode_lowercase(self):
        """Test Unicode character lowercasing."""
        assert lowercase("ÜBER") == "über"
        assert lowercase("MÜNCHEN") == "münchen"

    def test_function_attributes(self):
        """Test that function has correct attributes."""
        assert lowercase.token_only is False
        assert lowercase.length_preserving is True


class TestNfc:
    """Tests for the nfc() Unicode normalization function."""

    def test_basic_nfc(self):
        """Test basic NFC normalization."""
        # e + combining acute accent -> é
        decomposed = "e\u0301"  # e + combining acute accent
        composed = "é"
        assert nfc(decomposed) == composed

    def test_already_normalized(self):
        """Test text that is already NFC normalized."""
        assert nfc("hello") == "hello"
        assert nfc("café") == "café"

    def test_empty_string(self):
        """Test empty string input."""
        assert nfc("") == ""

    def test_function_attributes(self):
        """Test that function has correct attributes."""
        assert nfc.token_only is False
        assert nfc.length_preserving is False


class TestStripPunctuation:
    """Tests for the strip_punctuation() function."""

    def test_leading_punctuation(self):
        """Test stripping leading punctuation."""
        assert strip_punctuation("...hello") == "hello"
        assert strip_punctuation("'hello") == "hello"

    def test_trailing_punctuation(self):
        """Test stripping trailing punctuation."""
        assert strip_punctuation("hello...") == "hello"
        assert strip_punctuation("hello!") == "hello"

    def test_both_sides_punctuation(self):
        """Test stripping punctuation from both sides."""
        assert strip_punctuation("...hello!") == "hello"
        assert strip_punctuation("'test'") == "test"

    def test_internal_punctuation_preserved(self):
        """Test that internal punctuation is preserved."""
        assert strip_punctuation("don't") == "don't"
        assert strip_punctuation("hello-world") == "hello-world"

    def test_empty_string(self):
        """Test empty string input."""
        assert strip_punctuation("") == ""

    def test_punctuation_only(self):
        """Test string with only punctuation."""
        assert strip_punctuation("...") == ""

    def test_ampersand_preserved(self):
        """Test that & is preserved (not stripped)."""
        assert strip_punctuation("&hello") == "&hello"
        assert strip_punctuation("hello&") == "hello&"

    def test_percent_preserved(self):
        """Test that % is preserved (not stripped)."""
        assert strip_punctuation("%hello") == "%hello"
        assert strip_punctuation("hello%") == "hello%"

    def test_function_attributes(self):
        """Test that function has correct attributes."""
        assert strip_punctuation.token_only is True
        assert strip_punctuation.length_preserving is False


class TestTransliterateLatin:
    """Tests for the transliterate_latin_letters() function."""

    def test_accented_letters(self):
        """Test transliteration of accented Latin letters."""
        assert transliterate_latin_letters("é") == "e"
        assert transliterate_latin_letters("ñ") == "n"
        assert transliterate_latin_letters("ü") == "u"

    def test_mixed_text(self):
        """Test mixed text with some accented letters."""
        assert transliterate_latin_letters("café") == "cafe"
        assert transliterate_latin_letters("naïve") == "naive"

    def test_no_diacritics(self):
        """Test text without diacritics."""
        assert transliterate_latin_letters("hello") == "hello"

    def test_empty_string(self):
        """Test empty string input."""
        assert transliterate_latin_letters("") == ""

    def test_non_latin_preserved(self):
        """Test that non-Latin characters are preserved."""
        assert transliterate_latin_letters("日本") == "日本"
        assert transliterate_latin_letters("αβγ") == "αβγ"

    def test_function_attributes(self):
        """Test that function has correct attributes."""
        assert transliterate_latin_letters.token_only is False
        assert transliterate_latin_letters.length_preserving is False


class TestTransliterateSymbols:
    """Tests for the transliterate_symbols() function."""

    def test_basic_symbols(self):
        """Test transliteration of basic symbols."""
        # Smart quotes to regular quotes
        assert transliterate_symbols('"') == '"'
        assert transliterate_symbols("'") == "'"

    def test_letters_preserved(self):
        """Test that letters are preserved."""
        assert transliterate_symbols("hello") == "hello"

    def test_empty_string(self):
        """Test empty string input."""
        assert transliterate_symbols("") == ""

    def test_currency_symbols_preserved(self):
        """Test that currency symbols like € and £ are not transliterated."""
        assert transliterate_symbols("€") == "€"
        assert transliterate_symbols("£") == "£"

    def test_function_attributes(self):
        """Test that function has correct attributes."""
        assert transliterate_symbols.token_only is False
        assert transliterate_symbols.length_preserving is False


class TestRemoveSymbols:
    """Tests for the remove_symbols() function."""

    def test_remove_punctuation(self):
        """Test removal of punctuation."""
        assert remove_symbols("hello!") == "hello"
        assert remove_symbols("hello, world") == "hello world"

    def test_remove_symbols_chars(self):
        """Test removal of symbol characters."""
        assert remove_symbols("$100") == "100"

    def test_letters_preserved(self):
        """Test that letters are preserved."""
        assert remove_symbols("hello") == "hello"

    def test_numbers_preserved(self):
        """Test that numbers are preserved."""
        assert remove_symbols("123") == "123"

    def test_empty_string(self):
        """Test empty string input."""
        assert remove_symbols("") == ""

    def test_symbols_only(self):
        """Test string with only symbols."""
        assert remove_symbols("!!!") == ""
        assert remove_symbols("$$$") == ""

    def test_function_attributes(self):
        """Test that function has correct attributes."""
        assert remove_symbols.token_only is False
        assert remove_symbols.length_preserving is False


class TestNormalizer:
    """Tests for the Normalizer class."""

    def test_single_function_pipeline(self):
        """Test normalizer with a single function."""
        normalizer = Normalizer([(lowercase, {})], name="lowercase_only")
        assert normalizer("HELLO") == "hello"

    def test_multi_function_pipeline(self):
        """Test normalizer with multiple functions."""
        normalizer = Normalizer(
            [(lowercase, {}), (remove_symbols, {})],
            name="lowercase_and_remove",
        )
        assert normalizer("HELLO!") == "hello"

    def test_empty_pipeline(self):
        """Test normalizer with empty pipeline."""
        normalizer = Normalizer([], name="empty")
        assert normalizer("hello") == "hello"

    def test_repr_with_name(self):
        """Test string representation with name."""
        normalizer = Normalizer([(lowercase, {})], name="test")
        assert "test" in repr(normalizer)
        assert "lowercase" in repr(normalizer)

    def test_repr_no_name(self):
        """Test string representation without name."""
        normalizer = Normalizer([(lowercase, {})], name=None)
        assert "lowercase" in repr(normalizer)

    def test_pipeline_order(self):
        """Test that pipeline functions are applied in order."""
        # If we lowercase first, then transliterate, result differs from reverse
        normalizer = Normalizer(
            [(nfc, {}), (lowercase, {})],
            name="nfc_then_lower",
        )
        assert normalizer("CAFÉ") == "café"
