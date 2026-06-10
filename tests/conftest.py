"""Shared fixtures for BeWER unit tests."""

import pytest

from bewer.core.dataset import Dataset


class StubParent:
    """Lightweight stand-in for a parent in the src hierarchy.

    ``pipelines`` is now passed explicitly to core objects, so a parent is only
    needed where ``src`` is genuinely read. This stub stands in for that parent in
    unit tests that build a ``Token``/``Text`` in isolation; it exposes only ``raw``,
    which ``Token.inctx`` reads from its ``src``.
    """

    def __init__(self, *, raw=""):
        self.raw = raw


@pytest.fixture
def stub_parent():
    """A minimal parent object for constructing core objects in isolation."""
    return StubParent()


@pytest.fixture
def pipelines():
    """The resolved default pipeline registry, for constructing core objects standalone."""
    return Dataset().pipelines


@pytest.fixture
def sample_dataset():
    """Create a Dataset with a few ref/hyp pairs for testing."""
    dataset = Dataset()
    dataset.add("hello world", "hello world")
    dataset.add("the quick brown fox", "the quick brown dog")
    dataset.add("testing one two three", "testing one two")
    return dataset


@pytest.fixture
def sample_example(sample_dataset):
    """Create a single Example object for testing."""
    return sample_dataset[0]


@pytest.fixture
def sample_text(sample_example):
    """Create a Text object with known content for testing."""
    return sample_example.ref


@pytest.fixture
def sample_tokens(sample_text):
    """Create a TokenList with known tokens for testing."""
    return sample_text.tokens


@pytest.fixture
def empty_dataset():
    """Create an empty Dataset for testing edge cases."""
    return Dataset()


@pytest.fixture
def dataset_with_errors():
    """Create a Dataset where all hypotheses differ from references."""
    dataset = Dataset()
    dataset.add("hello", "goodbye")
    dataset.add("world", "earth")
    return dataset


@pytest.fixture
def dataset_perfect_match():
    """Create a Dataset where all hypotheses match references perfectly."""
    dataset = Dataset()
    dataset.add("hello world", "hello world")
    dataset.add("test phrase", "test phrase")
    return dataset
