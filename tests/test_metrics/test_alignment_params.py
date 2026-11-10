"""Tests for the shared `alignment` metric parameter (AlignmentParams)."""

import pytest

from bewer import Dataset, Vocabulary, get_config
from bewer.metrics.base import ALIGNMENT_ALGORITHMS, METRIC_REGISTRY

LEVENSHTEIN_DEFAULT = (
    "wer",
    "cer",
    "mer",
    "wip",
    "wil",
    "ktr",
    "ktp",
    "ktf",
    "ktfpr",
    "kter",
    "b_wer",
    "u_wer",
    "insertion_rate",
    "example_insertion_rate",
)
ERROR_ALIGN_DEFAULT = ("rktr", "ktcer")
NO_ALIGNMENT = ("summary", "per")


def _metadata(name):
    """Registry metadata for metric ``name``."""
    return METRIC_REGISTRY.metric_metadata[name]


@pytest.fixture
def dataset():
    """A dataset where error_align splits a compound (`bedroom`) and Levenshtein does not."""
    config = get_config("base").clone().add_vocabulary(Vocabulary("v", terms=["bed room"]))
    dataset = Dataset(config=config)
    dataset.add("the bed room is big", "the bedroom is a big")
    dataset.add("a bed room", "a bed room")
    dataset.add("hello world", "hello there world")
    return dataset


def _metric(dataset, name, **kwargs):
    """Build metric ``name`` on ``dataset``, adding the params it requires."""
    if "vocab" in _metadata(name)["param_schema"]:
        kwargs["vocab"] = "v"
    if name == "ktfpr":
        kwargs["partial_credit"] = True
    return dataset.metrics.get(name)(**kwargs)


class TestSupportedAlignments:
    """Each metric declares the alignments it accepts and its default."""

    @pytest.mark.parametrize("name", LEVENSHTEIN_DEFAULT + ERROR_ALIGN_DEFAULT)
    def test_supports_both_algorithms(self, name):
        """Test that alignment-based metrics accept both algorithms."""
        assert _metadata(name)["metric_cls"].supported_alignments == ALIGNMENT_ALGORITHMS

    @pytest.mark.parametrize("name", NO_ALIGNMENT)
    def test_metrics_without_alignment(self, name):
        """Test that summary and PER declare no alignments and take no alignment param."""
        assert _metadata(name)["metric_cls"].supported_alignments == ()
        assert "alignment" not in _metadata(name)["param_schema"]

    @pytest.mark.parametrize(
        ("name", "default"),
        [(name, "levenshtein") for name in LEVENSHTEIN_DEFAULT]
        + [(name, "error_align") for name in ERROR_ALIGN_DEFAULT],
    )
    def test_default(self, dataset, name, default):
        """Test each metric's default alignment."""
        assert _metric(dataset, name).params.alignment == default

    @pytest.mark.parametrize("alignment", ALIGNMENT_ALGORITHMS)
    @pytest.mark.parametrize("name", LEVENSHTEIN_DEFAULT + ERROR_ALIGN_DEFAULT)
    def test_computes_under_each_alignment(self, dataset, name, alignment):
        """Test that every alignment-based metric computes under each supported alignment."""
        assert isinstance(_metric(dataset, name, alignment=alignment).value, float)


class TestValidation:
    """Unsupported alignments are rejected at metric construction."""

    def test_unknown_alignment_raises(self, dataset):
        """Test that an unknown alignment raises ValueError naming the metric's supported ones."""
        with pytest.raises(ValueError, match="does not support alignment 'foo'"):
            dataset.metrics.wer(alignment="foo")

    @pytest.mark.parametrize("name", NO_ALIGNMENT)
    def test_metrics_without_alignment_reject_it(self, dataset, name):
        """Test that metrics without alignment reject the param."""
        with pytest.raises(ValueError):
            dataset.metrics.get(name)(alignment="levenshtein")


class TestConfidenceInterval:
    """Bootstrap CIs work for metrics computed from error_align."""

    def test_bootstrap_under_error_align(self, dataset):
        """Test that the CI brackets the error_align WER."""
        wer = dataset.metrics.wer(alignment="error_align")
        ci = wer.compute_confidence_interval(n_resamples=50, seed=0)
        assert ci.low <= wer.value <= ci.high
