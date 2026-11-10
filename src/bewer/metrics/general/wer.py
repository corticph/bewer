from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from bewer.metrics.base import (
    ALIGNMENT_ALGORITHMS,
    METRIC_REGISTRY,
    AlignmentParams,
    ExampleMetric,
    Metric,
    metric_property,
)

__all__ = ["WER"]


class WER_(ExampleMetric):
    @metric_property(is_value=True)
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference text.

        Under Levenshtein this is the word-level edit distance, computed directly. Under
        error_align it is the edit count of the error_align alignment, which can split one
        hypothesis token across ops or merge several into one op, so it can be higher or lower.
        """
        if self.params.alignment == "levenshtein":
            return Levenshtein.distance(
                self.example.ref.tokens.normalized,
                self.example.hyp.tokens.normalized,
            )
        return self.parent_metric._alignment[self.example.index].alignment.num_edits

    @metric_property(is_value=True)
    def ref_length(self) -> int:
        """Get the number of tokens in the reference text."""
        return len(self.example.ref.tokens)

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the example-level word error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length


@METRIC_REGISTRY.register("wer")
class WER(Metric):
    short_name_base = "WER"
    long_name_base = "Word Error Rate"
    description = (
        "Word error rate (WER) is computed as the word-level edit distance between the reference "
        "and hypothesis texts, divided by the total number of words in the reference texts. "
        "The `alignment` parameter (default 'levenshtein') selects the alignment the edits are counted "
        "from. 'error_align' can split a hypothesis word across operations or merge several into one, so its "
        "WER can be higher or lower than the Levenshtein WER."
    )
    example_cls = WER_
    param_schema = AlignmentParams
    supported_alignments = ALIGNMENT_ALGORITHMS

    @metric_property
    def _alignment(self):
        """The alignment metric selected by ``params.alignment``; only read for non-Levenshtein alignments."""
        return self._get_alignment_metric()

    @metric_property(is_value=True)
    def num_edits(self) -> int:
        """Get the number of edits between the hypothesis and reference texts."""
        return sum(em.num_edits for em in self)

    @metric_property(is_value=True)
    def ref_length(self) -> int:
        """Get the number of tokens in the reference texts."""
        return sum(em.ref_length for em in self)

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the word error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length
