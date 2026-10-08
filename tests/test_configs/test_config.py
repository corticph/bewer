"""Tests for bewer.config module — BewerConfig, configs, merge, resolution."""

import pytest

from bewer.config import (
    BewerConfig,
    Pipelines,
    Transform,
    get_config,
    merge_configs,
    resolve_config,
)
from bewer.preprocessing.normalization import lowercase, nfc, transliterate_latin_letters, transliterate_symbols


class TestBewerConfig:
    """Tests for the BewerConfig dataclass."""

    def test_frozen(self):
        """BewerConfig is frozen and cannot be mutated."""
        config = BewerConfig()
        with pytest.raises(AttributeError):
            config.standardizers = {}

    def test_replace(self):
        """replace() returns a copy with the given fields changed."""
        config = BewerConfig()
        new = config.replace(
            normalizers={
                "default": (Transform(lowercase),),
            },
        )
        assert "default" in new.normalizers
        assert len(new.normalizers["default"]) == 1
        assert config.normalizers == {}

    def test_default_factory_independent(self):
        """Each BewerConfig gets its own default dicts (no shared mutable default)."""
        a = BewerConfig()
        b = BewerConfig()
        a.standardizers["x"] = ()
        assert b.standardizers == {}


class TestMergeConfigs:
    """Tests for merge_configs()."""

    def test_variant_level_merge(self):
        """Delta variant replaces base variant entirely."""
        base = BewerConfig(
            normalizers={
                "default": (Transform(lowercase),),
                "cased": (Transform(transliterate_symbols),),
            },
        )
        delta = BewerConfig(
            normalizers={
                "default": (Transform(transliterate_latin_letters),),
            },
        )
        merged = merge_configs(base, delta)
        assert merged.normalizers["default"] == (Transform(transliterate_latin_letters),)
        assert merged.normalizers["cased"] == (Transform(transliterate_symbols),)

    def test_empty_delta_inherits_all(self):
        """An empty delta inherits everything from the base."""
        base = BewerConfig(
            standardizers={"default": (Transform(nfc),)},
        )
        merged = merge_configs(base, BewerConfig())
        assert merged.standardizers == base.standardizers


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
        steps = config.normalizers["default"]
        transliterate = [s for s in steps if s.fn == transliterate_latin_letters]
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
        steps = config.normalizers["default"]
        transliterate = [s for s in steps if s.fn == transliterate_latin_letters]
        assert transliterate[0].params.get("preserve") is not None

    def test_unknown_config_raises(self):
        """An unknown config raises ComponentNotFoundError."""
        from bewer.registry import ComponentNotFoundError

        with pytest.raises(ComponentNotFoundError, match="Config 'xx' not found"):
            get_config("xx")


class TestResolveConfigToPipelines:
    """Tests for resolve_config()."""

    def test_resolves_to_pipelines(self):
        """resolve_config returns a Pipelines namedtuple."""
        config = get_config("base")
        pipelines = resolve_config(config)
        assert isinstance(pipelines, Pipelines)

    def test_direct_callable_accepted(self):
        """A direct callable in a Transform is used as-is."""

        def my_upper(text: str) -> str:
            return text.upper()

        config = BewerConfig(
            normalizers={"default": (Transform(my_upper),)},
        )
        pipelines = resolve_config(config)
        assert "default" in pipelines.normalizers
