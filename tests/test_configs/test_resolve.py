"""Tests for bewer.configs.resolve module."""

import pytest
from omegaconf import OmegaConf

from bewer import Dataset
from bewer.configs.resolve import Pipelines, resolve_pipelines

STAGES = ("standardizers", "tokenizers", "normalizers")


class TestPipelinesRepr:
    """Tests for the Pipelines repr."""

    def test_lists_variant_names_per_stage(self):
        """Each stage gets its own line, listing the names registered under it."""
        text = repr(
            Pipelines(
                standardizers={"default": object()},
                tokenizers={"default": object(), "other": object()},
                normalizers={"default": object()},
            )
        )

        assert text.startswith("Pipelines(")
        assert text.endswith(")")
        assert "standardizers: default" in text
        assert "tokenizers:    default, other" in text
        assert "normalizers:   default" in text

    def test_empty_stage_renders_placeholder(self):
        """A stage with no variants renders as '-' rather than an empty gap."""
        text = repr(Pipelines(standardizers={}, tokenizers={"default": object()}, normalizers={}))

        assert "standardizers: -" in text
        assert "tokenizers:    default" in text

    def test_resolved_config_lists_a_default_per_stage(self):
        """A real configuration renders one line per stage, each offering a 'default'.

        Deliberately does not assert on any other variant name: those are config, not
        a property of the repr, and are free to change.
        """
        text = repr(Dataset(language="en").pipelines)
        lines = text.splitlines()

        assert len(lines) == 5  # header, three stages, closing paren
        for stage, line in zip(STAGES, lines[1:-1]):
            name, _, variants = line.partition(":")
            assert name.strip() == stage
            assert "default" in [variant.strip() for variant in variants.split(",")]

    def test_resolved_objects_are_not_inlined(self):
        """The repr names the variants; it does not dump the resolved pipeline objects."""
        text = repr(Dataset(language="en").pipelines)

        assert "Tokenizer(" not in text
        assert "Normalizer(" not in text

    def test_still_behaves_as_a_namedtuple(self):
        """The repr override must not disturb tuple semantics or field access."""
        pipelines = Pipelines(standardizers={"a": 1}, tokenizers={"b": 2}, normalizers={"c": 3})

        assert pipelines.standardizers == {"a": 1}
        assert tuple(pipelines) == ({"a": 1}, {"b": 2}, {"c": 3})
        assert pipelines._fields == STAGES
        assert getattr(pipelines, "tokenizers") == {"b": 2}


class TestResolveFuncPipeline:
    """Tests for _resolve_func_pipeline param validation."""

    def test_unexpected_param_raises(self):
        """An unexpected parameter in the config raises ValueError."""
        cfg = OmegaConf.create(
            {
                "standardizers": {"default": {"bewer.preprocessing.normalization.nfc": {"bogus": True}}},
                "tokenizers": {"default": {"bewer.preprocessing.tokenization.whitespace_strip_symbols_and_custom": {}}},
                "normalizers": {"default": {"bewer.preprocessing.normalization.lowercase": {}}},
            }
        )
        with pytest.raises(ValueError, match="Unexpected parameter 'bogus'"):
            resolve_pipelines(cfg)

    def test_first_positional_arg_not_in_config(self):
        """The first positional argument should not be passed in config params."""
        cfg = OmegaConf.create(
            {
                "standardizers": {"default": {"bewer.preprocessing.normalization.nfc": {"text": "hello"}}},
                "tokenizers": {"default": {"bewer.preprocessing.tokenization.whitespace_strip_symbols_and_custom": {}}},
                "normalizers": {"default": {"bewer.preprocessing.normalization.lowercase": {}}},
            }
        )
        with pytest.raises(ValueError, match="First positional argument 'text' should not be passed"):
            resolve_pipelines(cfg)


class TestResolveTokenizer:
    """Tests for _resolve_tokenizer validation."""

    def test_multiple_tokenizers_raises(self):
        """A tokenizer config with more than one definition raises ValueError."""
        cfg = OmegaConf.create(
            {
                "standardizers": {"default": {"bewer.preprocessing.normalization.nfc": {}}},
                "tokenizers": {
                    "default": {
                        "bewer.preprocessing.tokenization.whitespace_strip_symbols_and_custom": {},
                        "bewer.preprocessing.tokenization.keep_symbols_and_punctuation_pattern": {},
                    }
                },
                "normalizers": {"default": {"bewer.preprocessing.normalization.lowercase": {}}},
            }
        )
        with pytest.raises(ValueError, match="exactly one tokenizer"):
            resolve_pipelines(cfg)
