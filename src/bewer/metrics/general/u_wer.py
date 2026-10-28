from __future__ import annotations

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_property

from ._wer_stats_params import WERStatsParams

__all__ = ["UWER"]


class UWER_(ExampleMetric):
    @metric_property(is_value=True)
    def num_errors(self) -> int:
        """Errors attributed to the non-vocabulary (unbiased) side in this example."""
        return self.parent_metric._wer_stats[self.example.index].num_unbiased_errors

    @metric_property(is_value=True)
    def ref_length(self) -> int:
        """Reference tokens attributed to the unbiased side in this example."""
        return self.parent_metric._wer_stats[self.example.index].num_ref_other_tokens

    @metric_property(is_main=True)
    def value(self) -> float:
        """Example-level unbiased word error rate (U-WER)."""
        if self.ref_length == 0:
            return float(self.num_errors)
        return self.num_errors / self.ref_length


@METRIC_REGISTRY.register("u_wer")
class UWER(Metric):
    short_name_base = "U-WER"
    long_name_base = "Unbiased Word Error Rate"
    description = (
        "Unbiased word error rate (U-WER) is the word error rate on the words *not* in "
        "the biasing vocabulary (here: a `Vocabulary`), as defined by Le et al. "
        "(2021, arXiv:2104.02194) for contextual-biasing evaluation. Substitutions and "
        "deletions are attributed to U-WER when the reference token does not belong to "
        "the vocabulary; insertions when the hypothesis token does not. "
        "U-WER = unbiased_errors / num_ref_other_tokens, and together with B-WER it "
        "forms an exact two-way split of plain WER. The `multi_token` parameter "
        "(default 'occurrence') controls vocabulary membership of multi-token terms — "
        "see B-WER for the semantics of 'occurrence', 'flag', and 'split'."
    )
    example_cls = UWER_

    param_schema = WERStatsParams

    @metric_property(is_dependency=True)
    def _wer_stats(self):
        """Get the shared _WERStats metric instance."""
        return self.dataset.metrics._wer_stats(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            alignment=self.params.alignment,
            multi_token=self.params.multi_token,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_property(is_value=True)
    def num_errors(self) -> int:
        """Total errors attributed to the non-vocabulary (unbiased) side."""
        return self._wer_stats.num_unbiased_errors

    @metric_property(is_value=True)
    def ref_length(self) -> int:
        """Total reference tokens attributed to the unbiased side."""
        return self._wer_stats.num_ref_other_tokens

    @metric_property(is_main=True)
    def value(self) -> float:
        """Unbiased word error rate (U-WER)."""
        if self.ref_length == 0:
            return float(self.num_errors)
        return self.num_errors / self.ref_length
