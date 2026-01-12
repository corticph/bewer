from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from bewer.flags import DEFAULT
from bewer.metrics import example_helpers
from bewer.metrics.base import METRIC_REGISTRY, Metric

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset


@METRIC_REGISTRY.register("wer")
class WER(Metric):
    short_name = "WER"
    long_name = "Word Error Rate"
    description = (
        "Word Error Rate (WER) is computed as the token-level (i.e., word-level) edit distance between the reference "
        "and hypothesis texts, divided by the total number of tokens in the reference texts."
    )
    example_cls = example_helpers.WER

    def __init__(
        self,
        src: "Dataset",
        name: str = "wer",
        standardizer: str = DEFAULT,
        tokenizer: str = DEFAULT,
        normalizer: str = DEFAULT,
    ):
        """Initialize the WER Metric object."""
        self.pipeline = (standardizer, tokenizer, normalizer)
        super().__init__(src, name)

    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference texts."""
        return sum([example.metrics.__getattr__(self.name).num_edits for example in self._src_dataset])

    @cached_property
    def ref_length(self) -> int:
        """Get the number of tokens in the reference texts."""
        return sum([example.metrics.__getattr__(self.name).ref_length for example in self._src_dataset])

    @cached_property
    def value(self) -> float:
        """Get the word error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length


@METRIC_REGISTRY.register("cer")
class CER(Metric):
    short_name = "CER"
    long_name = "Character Error Rate"
    description = (
        "Character Error Rate (CER) is computed as the character-level edit distance between the normalized reference "
        "and hypothesis texts, divided by the total number of characters in the reference texts."
    )
    example_cls = example_helpers.CER

    def __init__(
        self,
        src: "Dataset",
        name: str = "cer",
        standardizer: str = DEFAULT,
        tokenizer: str = DEFAULT,
        normalizer: str = DEFAULT,
    ):
        """Initialize the CER Metric object."""
        self.pipeline = (standardizer, tokenizer, normalizer)
        super().__init__(src, name)

    @cached_property
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference texts."""
        return sum([example.metrics.get(self.name).num_edits for example in self._src_dataset])

    @cached_property
    def ref_length(self) -> int:
        """Get the number of characters in the reference texts."""
        return sum([example.metrics.get(self.name).ref_length for example in self._src_dataset])

    @cached_property
    def value(self) -> float:
        """Get the character error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length


@METRIC_REGISTRY.register("levenshtein")
class Levenshtein(Metric):
    short_name = "Levenshtein"
    long_name = "Levenshtein Alignment"
    description = "Levenshtein alignment between hypothesis and reference texts."
    example_cls = example_helpers.Levenshtein

    def __init__(
        self,
        src: "Dataset",
        name: str = "levenshtein",
        standardizer: str = DEFAULT,
        tokenizer: str = DEFAULT,
        normalizer: str = DEFAULT,
    ):
        """Initialize the Levenshtein Metric object."""
        self.pipeline = (standardizer, tokenizer, normalizer)
        super().__init__(src, name)
