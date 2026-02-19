"""Tests for bewer.core.caching module."""

from contextvars import ContextVar

import pytest

from bewer.core.caching import _STAGE_INDEX, PipelineCachedProperty, pipeline_cached_property
from bewer.preprocessing.context import (
    PIPELINE_STAGES,
    STANDARDIZER_NAME,
    set_pipeline,
)


class TestStageIndex:
    """Tests for the _STAGE_INDEX lookup built at import time."""

    def test_all_stages_registered(self):
        """Every context var in PIPELINE_STAGES has an entry."""
        for cv, _ in PIPELINE_STAGES:
            assert cv in _STAGE_INDEX

    def test_first_stage_has_single_context_var(self):
        """The first pipeline stage key contains only itself."""
        first_cv, first_attr = PIPELINE_STAGES[0]
        context_vars, attr = _STAGE_INDEX[first_cv]
        assert context_vars == (first_cv,)
        assert attr == first_attr

    def test_later_stages_include_preceding_vars(self):
        """Each stage's key tuple includes all preceding stages."""
        for i, (cv, attr) in enumerate(PIPELINE_STAGES):
            context_vars, resolved_attr = _STAGE_INDEX[cv]
            expected = tuple(c for c, _ in PIPELINE_STAGES[: i + 1])
            assert context_vars == expected
            assert resolved_attr == attr


class TestPipelineCachedPropertyDescriptor:
    """Tests for the PipelineCachedProperty descriptor directly."""

    def test_class_access_returns_descriptor(self):
        """Accessing the descriptor on the class returns the descriptor itself."""
        STAGE = ContextVar("STAGE", default="default")
        _STAGE_INDEX[STAGE] = ((STAGE,), "funcs")

        class Dummy:
            _cache_value = {}
            _pipelines = None

            @pipeline_cached_property(STAGE)
            def value(self, func):
                return func()

        assert isinstance(Dummy.__dict__["value"], PipelineCachedProperty)
        del _STAGE_INDEX[STAGE]

    def test_set_name_sets_cache_attr(self):
        """__set_name__ derives the cache attribute name correctly."""
        prop = PipelineCachedProperty(
            context_vars=(STANDARDIZER_NAME,),
            pipeline="standardizers",
            compute=lambda self, fn: fn(self.raw),
        )
        prop.__set_name__(object, "standardized")
        assert prop._name == "standardized"
        assert prop._cache_attr == "_cache_standardized"


class TestPipelineCachedPropertyIntegration:
    """Integration tests for pipeline_cached_property using set_pipeline context."""

    def test_cache_hit_returns_same_object(self, sample_text):
        """Accessing the same property twice returns the cached result."""
        std1 = sample_text.standardized
        std2 = sample_text.standardized
        assert std1 is std2

    def test_different_pipeline_produces_different_result(self, sample_dataset):
        """Switching the pipeline context var produces a fresh computation."""
        text = sample_dataset[0].ref  # "hello world"

        std_default = text.standardized

        # Register an uppercase standardizer under a new name
        text._pipelines.standardizers["upper"] = str.upper

        with set_pipeline(standardizer="upper"):
            std_upper = text.standardized

        assert std_default != std_upper
        assert std_upper == "HELLO WORLD"

    def test_cache_is_per_instance(self, sample_dataset):
        """Each Text instance has its own cache dict."""
        text_a = sample_dataset[0].ref
        text_b = sample_dataset[1].ref
        _ = text_a.standardized
        _ = text_b.standardized
        assert text_a._cache_standardized is not text_b._cache_standardized

    def test_raises_when_pipelines_is_none(self):
        """Raises ValueError when _pipelines is None."""
        from bewer.core.text import Text

        text = Text(raw="hello")
        with pytest.raises(ValueError, match="No standardizers found"):
            _ = text.standardized

    def test_raises_when_pipeline_name_not_found(self, sample_dataset):
        """Raises ValueError when the context var names a missing pipeline entry."""
        text = sample_dataset[0].ref

        with set_pipeline(standardizer="nonexistent"):
            with pytest.raises(ValueError, match="'nonexistent' not found"):
                _ = text.standardized

    def test_token_normalized_uses_normalizer_stage(self, sample_dataset):
        """Token.normalized goes through the normalizer pipeline stage."""
        text = sample_dataset[0].ref
        token = text.tokens[0]  # "hello"
        normalized = token.normalized
        assert isinstance(normalized, str)

    def test_token_normalized_different_normalizers(self, sample_dataset):
        """Switching normalizer context produces different normalized values."""
        text = sample_dataset[0].ref
        token = text.tokens[0]

        norm_default = token.normalized

        token._pipelines.normalizers["shout"] = lambda s: s.upper() + "!"

        with set_pipeline(normalizer="shout"):
            norm_shout = token.normalized

        assert norm_default != norm_shout
        assert norm_shout == "HELLO!"

    def test_tokens_cache_key_includes_standardizer_and_tokenizer(self, sample_dataset):
        """The tokens cache key is a tuple of (standardizer, tokenizer) names."""
        text = sample_dataset[0].ref
        _ = text.tokens
        # Default pipeline names
        assert ("default", "default") in text._cache_tokens

    def test_standardized_cache_key_is_single_value(self, sample_dataset):
        """The standardized cache key is a single string (not a tuple)."""
        text = sample_dataset[0].ref
        _ = text.standardized
        assert "default" in text._cache_standardized
