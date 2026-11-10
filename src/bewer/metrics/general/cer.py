from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from bewer.alignment import Alignment
from bewer.metrics.base import (
    ALIGNMENT_ALGORITHMS,
    METRIC_REGISTRY,
    AlignmentParams,
    ExampleMetric,
    Metric,
    metric_property,
)

__all__ = ["CER"]


def _alignment_char_edits(alignment: Alignment) -> int:
    """Count character edits along a word alignment, including the spaces between tokens.

    Each op costs the character edit distance between its reference and hypothesis text (the
    full text length for an insertion or deletion). A side has a space before an op when that
    op's text starts after the previous op's text on that side ends (a gap in the character
    spans). A piece of a hypothesis token split across several ops (error_align) continues
    without a gap. A space present on one side only costs one edit. The result is the cost of
    a valid edit script, so it is never below the direct character edit distance between the
    joined texts.
    """
    edits = 0
    ref_spaces: set[int] = set()
    hyp_spaces: set[int] = set()
    prev_ref_stop = prev_hyp_stop = None
    for i, op in enumerate(alignment):
        edits += Levenshtein.distance(op.ref or "", op.hyp or "")
        if op.ref_span is not None:
            if prev_ref_stop is not None and op.ref_span.start > prev_ref_stop:
                ref_spaces.add(i)
            prev_ref_stop = op.ref_span.stop
        if op.hyp_span is not None:
            if prev_hyp_stop is not None and op.hyp_span.start > prev_hyp_stop:
                hyp_spaces.add(i)
            prev_hyp_stop = op.hyp_span.stop
    return edits + len(ref_spaces ^ hyp_spaces)


class CER_(ExampleMetric):
    @metric_property(is_value=True)
    def num_edits(self) -> int:
        """Get the number of character edits between the hypothesis and reference text.

        Under Levenshtein this is the character-level edit distance between the joined texts,
        computed directly. Under error_align it is counted along the word alignment (see
        :func:`_alignment_char_edits`) and is never below the Levenshtein count.
        """
        if self.params.alignment == "levenshtein":
            return Levenshtein.distance(
                self.example.hyp.joined(),
                self.example.ref.joined(),
            )
        return _alignment_char_edits(self.parent_metric._alignment[self.example.index].alignment)

    @metric_property(is_value=True)
    def ref_length(self) -> int:
        """Get the number of characters in the reference text."""
        return len(self.example.ref.joined())

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the example-level character error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length


@METRIC_REGISTRY.register("cer")
class CER(Metric):
    short_name_base = "CER"
    long_name_base = "Character Error Rate"
    description = (
        "Character error rate (CER) is computed as the character-level edit distance between the reference "
        "and hypothesis texts, divided by the total number of characters in the reference texts. "
        "The `alignment` parameter (default 'levenshtein') selects how edits are counted: 'levenshtein' "
        "is the character-level edit distance; 'error_align' counts character edits within each pair of "
        "aligned words, plus one per unmatched space between words, so it never falls below the Levenshtein CER."
    )
    example_cls = CER_
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
        """Get the number of characters in the reference texts."""
        return sum(em.ref_length for em in self)

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the character error rate."""
        if self.ref_length == 0:
            return float(self.num_edits)
        return self.num_edits / self.ref_length
