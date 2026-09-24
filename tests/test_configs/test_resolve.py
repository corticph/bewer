"""Tests for bewer.configs.resolve module."""

from bewer import Dataset
from bewer.configs.resolve import Pipelines


class TestPipelinesRepr:
    """Tests for the Pipelines repr."""

    def test_lists_variant_names_per_stage(self):
        """The repr names each stage's variants rather than dumping the resolved objects."""
        text = repr(Dataset(language="en").pipelines)

        assert text.startswith("Pipelines(")
        assert text.endswith(")")
        assert "standardizers: default" in text
        assert "key_term" in text
        assert "normalizers:" in text and "cased" in text

    def test_is_multiline_and_compact(self):
        """One line per stage, and no resolved Tokenizer/Normalizer objects inlined."""
        text = repr(Dataset(language="en").pipelines)

        assert len(text.splitlines()) == 5  # header, three stages, closing paren
        assert "Tokenizer(" not in text
        assert "Normalizer(" not in text

    def test_empty_stage_renders_placeholder(self):
        """A stage with no variants renders as '-' rather than an empty gap."""
        text = repr(Pipelines(standardizers={}, tokenizers={"default": object()}, normalizers={}))

        assert "standardizers: -" in text
        assert "tokenizers:    default" in text

    def test_still_behaves_as_a_namedtuple(self):
        """The repr override must not disturb tuple semantics or field access."""
        pipelines = Pipelines(standardizers={"a": 1}, tokenizers={"b": 2}, normalizers={"c": 3})

        assert pipelines.standardizers == {"a": 1}
        assert tuple(pipelines) == ({"a": 1}, {"b": 2}, {"c": 3})
        assert pipelines._fields == ("standardizers", "tokenizers", "normalizers")
        assert getattr(pipelines, "tokenizers") == {"b": 2}
