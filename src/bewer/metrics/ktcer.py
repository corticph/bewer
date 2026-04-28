from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value

__all__ = ["KTCER"]


class KTCER_(ExampleMetric):
    @metric_value
    def num_char_edits(self) -> int:
        """Get the total character edits across all key term occurrences in the example."""
        return sum(ts.char_edits for ts in self.parent_metric._rkt_stats.get_example_metric(self.example).term_stats)

    @metric_value
    def ref_chars(self) -> int:
        """Get the total reference character count across all key term occurrences in the example."""
        return sum(ts.ref_chars for ts in self.parent_metric._rkt_stats.get_example_metric(self.example).term_stats)

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level key term character error rate."""
        if self.ref_chars == 0:
            return float(self.num_char_edits)
        return self.num_char_edits / self.ref_chars


@METRIC_REGISTRY.register("ktcer", tokenizer="key_term")
class KTCER(Metric):
    short_name_base = "KTCER"
    long_name_base = "Key Term Character Error Rate"
    description = (
        "Key term character error rate (KTCER) is computed as the total character-level edit distance "
        "across all key term occurrences in the reference, divided by the total number of reference "
        "characters in those key terms. Alignment is derived from error_align. Complements KTR by "
        "providing a continuous measure of transcription accuracy for domain-critical terminology."
    )
    example_cls = KTCER_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the KTCER metric.

        Attributes:
            vocab: The vocabulary name to use for key term identification.
            normalized: Whether to use normalized tokens for alignment and key term matching.
            allow_subset_matches: Whether to allow subset matches.
            only_local_matches: If True, restrict matching to per-example local key terms only.
        """

        vocab: str
        normalized: bool = True
        allow_subset_matches: bool = False
        only_local_matches: bool = False

        def validate(self) -> None:
            is_global_vocab = self.vocab in self.metric.dataset._global_key_term_vocabs
            is_local_vocab = self.vocab in self.metric.dataset._local_key_term_vocabs
            if not is_global_vocab and not is_local_vocab:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset key term vocabularies.")

    @dependency
    def _rkt_stats(self):
        """Get the shared _RKTStats metric instance."""
        return self.dataset.metrics._rkt_stats(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            allow_subset_matches=self.params.allow_subset_matches,
            only_local_matches=self.params.only_local_matches,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_value
    def num_char_edits(self) -> int:
        """Get the total character edits across all key term occurrences."""
        return sum(self.get_example_metric(example).num_char_edits for example in self._src)

    @metric_value
    def ref_chars(self) -> int:
        """Get the total reference character count across all key term occurrences."""
        return sum(self.get_example_metric(example).ref_chars for example in self._src)

    @metric_value(main=True)
    def value(self) -> float:
        """Get the key term character error rate."""
        if self.ref_chars == 0:
            return float(self.num_char_edits)
        return self.num_char_edits / self.ref_chars
