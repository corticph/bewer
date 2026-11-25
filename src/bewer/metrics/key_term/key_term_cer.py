from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, dependency, metric_property

from ._keyterm_params import KeyTermSubsetParams

__all__ = ["KTCER"]


class KTCER_(ExampleMetric):
    @metric_property(type="value")
    def num_char_edits(self) -> int:
        """Get the total character edits across all key term occurrences in the example."""
        return sum(ts.char_edits for ts in self.parent_metric._rkt_outcomes[self.example.index].term_outcomes)

    @metric_property(type="value")
    def ref_chars(self) -> int:
        """Get the total reference character count across all key term occurrences in the example."""
        return sum(ts.ref_chars for ts in self.parent_metric._rkt_outcomes[self.example.index].term_outcomes)

    @metric_property(type="main")
    def value(self) -> float:
        """Get the example-level key term character error rate."""
        if self.ref_chars == 0:
            return float(self.num_char_edits)
        return self.num_char_edits / self.ref_chars


@METRIC_REGISTRY.register("key_term_cer", tokenizer="key_term")
class KTCER(Metric):
    short_name_base = "KTCER"
    long_name_base = "Key-Term Character Error Rate"
    description = (
        "Key term character error rate (KTCER) is computed as the total character-level edit distance "
        "across all key term occurrences in the reference, divided by the total number of reference "
        "characters in those key terms. Alignment is derived from error_align. Complements KTR by "
        "providing a continuous measure of transcription accuracy for domain-critical terminology."
    )
    example_cls = KTCER_
    _rkt_outcomes = dependency("_rkt_outcomes", forward=("vocab", "allow_subset_matches"))

    @dataclass
    class param_schema(KeyTermSubsetParams):
        """Parameters for the KTCER metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            allow_subset_matches: Whether to allow subset matches.
        """

    @metric_property(type="value")
    def num_char_edits(self) -> int:
        """Get the total character edits across all key term occurrences."""
        return sum(em.num_char_edits for em in self)

    @metric_property(type="value")
    def ref_chars(self) -> int:
        """Get the total reference character count across all key term occurrences."""
        return sum(em.ref_chars for em in self)

    @metric_property(type="main")
    def value(self) -> float:
        """Get the key term character error rate."""
        if self.ref_chars == 0:
            return float(self.num_char_edits)
        return self.num_char_edits / self.ref_chars
