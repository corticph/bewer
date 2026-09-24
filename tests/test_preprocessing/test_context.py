"""Tests for bewer.preprocessing.context module."""

import bewer
from bewer import Dataset, set_pipeline
from bewer.preprocessing import set_pipeline as set_pipeline_from_subpackage
from bewer.preprocessing.context import set_pipeline as set_pipeline_from_module


class TestSetPipelineExport:
    """set_pipeline is public API and reachable without importing a private module."""

    def test_exported_from_top_level(self):
        assert set_pipeline is set_pipeline_from_module
        assert "set_pipeline" in bewer.__all__

    def test_exported_from_preprocessing_subpackage(self):
        assert set_pipeline_from_subpackage is set_pipeline_from_module
        assert "set_pipeline" in bewer.preprocessing.__all__


class TestSetPipelineContext:
    """Switching the active pipeline changes what the same Text resolves to."""

    def test_tokenizer_switch_applies_and_reverts(self):
        """The key_term tokenizer splits possessives, so "Crohn's" becomes two tokens."""
        dataset = Dataset(language="en")
        dataset.add(ref="Crohn's disease", hyp="Crohn's disease")
        text = dataset[0].ref

        assert text.tokens.normalized == ["crohn's", "disease"]

        with set_pipeline(tokenizer="key_term"):
            assert text.tokens.normalized == ["crohn", "s", "disease"]

        assert text.tokens.normalized == ["crohn's", "disease"]

    def test_normalizer_switch_applies_and_reverts(self):
        dataset = Dataset(language="en")
        dataset.add(ref="An MRI", hyp="An MRI")
        text = dataset[0].ref

        assert text.tokens.normalized == ["an", "mri"]

        with set_pipeline(normalizer="cased"):
            assert text.tokens.normalized == ["An", "MRI"]

        assert text.tokens.normalized == ["an", "mri"]
