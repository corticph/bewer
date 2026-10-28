from __future__ import annotations

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_property

from ._wer_stats_params import WERStatsParams

__all__ = ["BWER"]


class BWER_(ExampleMetric):
    @metric_property(is_value=True)
    def num_errors(self) -> int:
        """Errors attributed to the vocabulary (biased) side in this example."""
        return self.parent_metric._wer_stats[self.example.index].num_biased_errors

    @metric_property(is_value=True)
    def ref_length(self) -> int:
        """Reference tokens attributed to the biased side in this example."""
        return self.parent_metric._wer_stats[self.example.index].num_ref_vocab_tokens

    @metric_property(is_main=True)
    def value(self) -> float:
        """Example-level biased word error rate (B-WER)."""
        if self.ref_length == 0:
            return float(self.num_errors)
        return self.num_errors / self.ref_length


@METRIC_REGISTRY.register("b_wer")
class BWER(Metric):
    short_name_base = "B-WER"
    long_name_base = "Biased Word Error Rate"
    description = (
        "Biased word error rate (B-WER) is the word error rate restricted to words in "
        "the biasing vocabulary (here: a `Vocabulary`), as defined by Le et al. "
        "(2021, arXiv:2104.02194) for contextual-biasing evaluation. Substitutions and "
        "deletions are attributed to B-WER when the reference token belongs to the "
        "vocabulary; insertions when the hypothesis token does. "
        "B-WER = biased_errors / num_ref_vocab_tokens, and together with U-WER it forms "
        "an exact two-way split of plain WER. The `multi_token` parameter (default "
        "'occurrence') controls vocabulary membership of multi-token terms: 'occurrence' "
        "and 'flag' count a token as biased only when it is part of a matched term "
        "occurrence ('flag' also warns when the vocabulary contains terms that tokenize "
        "to multiple tokens); 'split' treats each term token as an individual "
        "in-vocabulary word."
    )
    example_cls = BWER_

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
        """Total errors attributed to the vocabulary (biased) side."""
        return self._wer_stats.num_biased_errors

    @metric_property(is_value=True)
    def ref_length(self) -> int:
        """Total reference tokens attributed to the biased side."""
        return self._wer_stats.num_ref_vocab_tokens

    @metric_property(is_main=True)
    def value(self) -> float:
        """Biased word error rate (B-WER)."""
        if self.ref_length == 0:
            return float(self.num_errors)
        return self.num_errors / self.ref_length
