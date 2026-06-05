"""Shared fixtures for BeWER unit tests."""

import pytest

from bewer.core.dataset import Dataset


class StubParent:
    """Lightweight stand-in for a parent in the src hierarchy.

    Now that ``src`` is required at construction time, unit tests that build a
    ``Token``/``Text``/``TokenList`` in isolation (without a real ``Dataset``)
    use this stub as the parent. It exposes only the attributes children read:
    ``pipelines`` (consumed by the pipeline caching descriptor) and ``raw``.
    """

    def __init__(self, *, pipelines=None, raw=""):
        self.pipelines = pipelines
        self.raw = raw


@pytest.fixture
def stub_parent():
    """A minimal parent object for constructing core objects in isolation."""
    return StubParent()


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
