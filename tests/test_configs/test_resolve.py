"""Tests for bewer.configs.resolve module."""

from bewer import Dataset
from bewer.configs.resolve import Pipelines

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
