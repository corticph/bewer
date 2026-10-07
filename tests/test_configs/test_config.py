"""Tests for bewer.config module — BewerConfig, profiles, merge, resolution."""

import pytest

from bewer.config import (
    BewerConfig,
    Pipelines,
    PipelineStep,
    from_yaml,
    merge_configs,
    resolve_config,
    resolve_profile,
    to_yaml,
)


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


class TestResolveProfile:
    """Tests for profile resolution."""

    def test_base_profile(self):
        """The base profile resolves to all expected variants."""
        config = resolve_profile("base")
        assert "default" in config.standardizers
        assert "default" in config.tokenizers
        assert "cased" in config.normalizers
        assert "with_punctuation" in config.tokenizers

    def test_en_profile_equals_base(self):
        """English profile is an empty delta — equals base."""
        config = resolve_profile("en")
        base = resolve_profile("base")
        assert config.standardizers == base.standardizers
        assert config.tokenizers == base.tokenizers
        assert config.normalizers == base.normalizers

    def test_da_profile_changes_normalizer(self):
        """Danish profile changes the default normalizer."""
        config = resolve_profile("da")
        steps = config.normalizers["default"]
        transliterate = [s for s in steps if s.component == "transliterate_latin_letters"]
        assert len(transliterate) == 1
        assert transliterate[0].params.get("preserve") == "\u00e6\u00f8\u00e5"

    def test_da_profile_inherits_tokenizers(self):
        """Danish profile inherits tokenizers from base."""
        config = resolve_profile("da")
        base = resolve_profile("base")
        assert config.tokenizers == base.tokenizers

    def test_fr_profile_changes_tokenizer_and_normalizer(self):
        """French profile changes both tokenizer and normalizer."""
        config = resolve_profile("fr")
        assert config.tokenizers["default"].params["split_on_escaped"] == "-/'"
        steps = config.normalizers["default"]
        transliterate = [s for s in steps if s.component == "transliterate_latin_letters"]
        assert transliterate[0].params.get("preserve") is not None

    def test_unknown_profile_raises(self):
        """An unknown profile raises ComponentNotFoundError."""
        from bewer.registry import ComponentNotFoundError

        with pytest.raises(ComponentNotFoundError, match="Profile 'xx' not found"):
            resolve_profile("xx")


class TestResolveConfig:
    """Tests for resolve_config()."""

    def test_resolves_to_pipelines(self):
        """resolve_config returns a Pipelines namedtuple."""
        config = resolve_profile("base")
        pipelines = resolve_config(config)
        assert isinstance(pipelines, Pipelines)

    def test_unknown_component_raises(self):
        """An unknown component name raises ComponentNotFoundError."""
        from bewer.registry import ComponentNotFoundError

        config = BewerConfig(
            normalizers={"default": (PipelineStep("nonexistent"),)},
            tokenizers={},
            standardizers={},
        )
        with pytest.raises(ComponentNotFoundError, match="Transform 'nonexistent' not found"):
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
        """to_yaml then from_yaml produces an equal config for the base profile."""
        config = resolve_profile("base")
        yaml_str = to_yaml(config)
        restored = from_yaml(yaml_str)
        assert restored.standardizers == config.standardizers
        assert restored.tokenizers == config.tokenizers
        assert restored.normalizers == config.normalizers

    def test_round_trip_da(self):
        """to_yaml then from_yaml works for Danish profile."""
        config = resolve_profile("da")
        yaml_str = to_yaml(config)
        restored = from_yaml(yaml_str)
        assert restored.normalizers["default"] == config.normalizers["default"]

    def test_callable_not_serializable(self):
        """A config with direct callables raises SerializationError."""
        from bewer.config import SerializationError

        def my_upper(text: str) -> str:
            return text.upper()

        config = BewerConfig(normalizers={"default": (PipelineStep(my_upper),)})
        with pytest.raises(SerializationError):
            to_yaml(config)
