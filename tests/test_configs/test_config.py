"""Tests for bewer.configs module — BewerConfig, configs, resolution."""

import pytest

from bewer.configs import (
    BewerConfig,
    Transform,
    get_config,
)
from bewer.preprocessing.normalization import (
    Normalizer,
    lowercase,
    nfc,
    transliterate_latin_letters,
    transliterate_symbols,
)


class TestBewerConfig:
    """Tests for the BewerConfig class."""

    def test_default_factory_independent(self):
        """Each BewerConfig gets its own default dicts (no shared mutable default)."""
        a = BewerConfig()
        b = BewerConfig()
        a.standardizers["x"] = None
        assert b.standardizers == {}


class TestExtends:
    """Tests for BewerConfig extends."""

    def test_variant_level_merge(self):
        """Delta variant replaces base variant entirely."""
        base = BewerConfig(
            normalizers=(
                Normalizer("default", Transform(lowercase)),
                Normalizer("cased", Transform(transliterate_symbols)),
            ),
        )
        delta = BewerConfig(
            extends=base,
            normalizers=(Normalizer("default", Transform(transliterate_latin_letters)),),
        )
        assert delta.normalizers["default"] == Normalizer("default", Transform(transliterate_latin_letters))
        assert delta.normalizers["cased"] == Normalizer("cased", Transform(transliterate_symbols))

    def test_empty_delta_inherits_all(self):
        """An empty delta inherits everything from the base."""
        base = BewerConfig(
            standardizers=(Normalizer("default", Transform(nfc)),),
        )
        delta = BewerConfig(extends=base)
        assert delta.standardizers == base.standardizers


class TestResolveConfig:
    """Tests for config resolution."""

    def test_base_config(self):
        """The base config resolves to all expected variants."""
        config = get_config("base")
        assert "default" in config.standardizers
        assert "default" in config.tokenizers
        assert "cased" in config.normalizers
        assert "with_punctuation" in config.tokenizers

    def test_en_config_equals_base(self):
        """English config is an empty delta — equals base."""
        config = get_config("en")
        base = get_config("base")
        assert config.standardizers == base.standardizers
        assert config.tokenizers == base.tokenizers
        assert config.normalizers == base.normalizers

    def test_da_config_changes_normalizer(self):
        """Danish config changes the default normalizer."""
        config = get_config("da")
        normalizer = config.normalizers["default"]
        transliterate = [t for t in normalizer.transforms if t.fn == transliterate_latin_letters]
        assert len(transliterate) == 1
        assert transliterate[0].params.get("preserve") == "æøå"

    def test_da_config_inherits_tokenizers(self):
        """Danish config inherits tokenizers from base."""
        config = get_config("da")
        base = get_config("base")
        assert config.tokenizers == base.tokenizers

    def test_fr_config_changes_tokenizer_and_normalizer(self):
        """French config changes both tokenizer and normalizer."""
        config = get_config("fr")

        fr_pattern = config.tokenizers["default"]
        base_pattern = get_config("base").tokenizers["default"]
        assert fr_pattern.pattern != base_pattern.pattern
        normalizer = config.normalizers["default"]
        transliterate = [t for t in normalizer.transforms if t.fn == transliterate_latin_letters]
        assert transliterate[0].params.get("preserve") is not None

    def test_unknown_config_raises(self):
        """An unknown config raises KeyError."""
        with pytest.raises(KeyError, match="Config 'xx' not found"):
            get_config("xx")


class TestDirectCallable:
    """Tests for direct callable acceptance."""

    def test_direct_callable_accepted(self):
        """A direct callable in a Transform is used as-is."""

        def my_upper(text: str) -> str:
            return text.upper()

        config = BewerConfig(
            normalizers=(Normalizer("default", Transform(my_upper)),),
        )
        assert "default" in config.normalizers
