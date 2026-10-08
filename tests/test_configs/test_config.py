"""Tests for bewer.config module — BewerConfig, configs, merge, resolution."""

import pytest

from bewer.config import (
    BewerConfig,
    Pipelines,
    PipelineStep,
    from_yaml,
    get_config,
    merge_configs,
    resolve_config,
    to_yaml,
)
from bewer.preprocessing.normalization import transliterate_latin_letters


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
                "default": (PipelineStep("lowercase"),),
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
                "default": (PipelineStep("lowercase"),),
                "cased": (PipelineStep("transliterate_symbols"),),
            },
        )
        delta = BewerConfig(
            normalizers={
                "default": (PipelineStep("transliterate_latin_letters"),),
            },
        )
        merged = merge_configs(base, delta)
        assert merged.normalizers["default"] == (PipelineStep("transliterate_latin_letters"),)
        assert merged.normalizers["cased"] == (PipelineStep("transliterate_symbols"),)

    def test_empty_delta_inherits_all(self):
        """An empty delta inherits everything from the base."""
        base = BewerConfig(
            standardizers={"default": (PipelineStep("nfc"),)},
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
        transliterate = [s for s in steps if s.component == transliterate_latin_letters]
        assert len(transliterate) == 1
        assert transliterate[0].params.get("preserve") == "\u00e6\u00f8\u00e5"

    def test_da_config_inherits_tokenizers(self):
        """Danish config inherits tokenizers from base."""
        config = get_config("da")
        base = get_config("base")
        assert config.tokenizers == base.tokenizers

    def test_fr_config_changes_tokenizer_and_normalizer(self):
        """French config changes both tokenizer and normalizer."""
        config = get_config("fr")
        assert config.tokenizers["default"].params["split_on_escaped"] == "-/'"
        steps = config.normalizers["default"]
        transliterate = [s for s in steps if s.component == transliterate_latin_letters]
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

    def test_unknown_component_raises(self):
        """A bare component name (no dots) raises ValueError."""
        config = BewerConfig(
            normalizers={"default": (PipelineStep("nonexistent"),)},
            tokenizers={},
            standardizers={},
        )
        with pytest.raises(ValueError, match="not a dotted path"):
            resolve_config(config)

    def test_direct_callable_accepted(self):
        """A direct callable in a PipelineStep is used as-is."""

        def my_upper(text: str) -> str:
            return text.upper()

        config = BewerConfig(
            normalizers={"default": (PipelineStep(my_upper),)},
        )
        pipelines = resolve_config(config)
        assert "default" in pipelines.normalizers


class TestYamlRoundTrip:
    """Tests for to_yaml / from_yaml."""

    def test_round_trip_base(self):
        """to_yaml then from_yaml then to_yaml produces the same YAML for base config."""
        config = get_config("base")
        yaml_str = to_yaml(config)
        restored = from_yaml(yaml_str)
        assert to_yaml(restored) == yaml_str

    def test_round_trip_da(self):
        """to_yaml then from_yaml then to_yaml produces the same YAML for Danish config."""
        config = get_config("da")
        yaml_str = to_yaml(config)
        restored = from_yaml(yaml_str)
        assert to_yaml(restored) == yaml_str

    def test_callable_not_serializable(self):
        """A config with direct callables raises SerializationError."""
        from bewer.config import SerializationError

        def my_upper(text: str) -> str:
            return text.upper()

        config = BewerConfig(normalizers={"default": (PipelineStep(my_upper),)})
        with pytest.raises(SerializationError):
            to_yaml(config)
