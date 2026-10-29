from __future__ import annotations

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_property

from ._keyterm_params import KeyTermParams

__all__ = ["BWER"]


class BWER_(ExampleMetric):
    @metric_property(is_value=True)
    def num_biased_edits(self) -> int:
        """Edits attributed to the biased (vocabulary) side in this example."""
        return self.parent_metric._bu_wer_outcomes[self.example.index].num_biased_edits

    @metric_property(is_value=True)
    def num_biased_ref_tokens(self) -> int:
        """Reference tokens attributed to the biased side in this example."""
        return self.parent_metric._bu_wer_outcomes[self.example.index].num_biased_ref_tokens

    @metric_property(is_main=True)
    def value(self) -> float:
        """Example-level biased word error rate (B-WER)."""
        if self.num_biased_ref_tokens == 0:
            return float(self.num_biased_edits)
        return self.num_biased_edits / self.num_biased_ref_tokens


@METRIC_REGISTRY.register("b_wer")
class BWER(Metric):
    short_name_base = "B-WER"
    long_name_base = "Biased Word Error Rate"
    description = (
        "Biased word error rate (B-WER) is the word error rate restricted to words in "
        "the biasing vocabulary (here: a `Vocabulary`), as defined by Le et al. "
        "(2021, arXiv:2104.02194) for contextual-biasing evaluation. Substitutions and "
        "deletions are attributed to B-WER when the reference token belongs to the "
        "vocabulary. Insertions are attributed when the hypothesis token does. "
        "B-WER = num_biased_edits / num_biased_ref_tokens, and together with U-WER it forms "
        "an exact two-way split of plain WER. A word counts as in-vocabulary only when "
        "its token belongs to a matched occurrence of a vocabulary term. The words of "
        'multi-token terms such as "blood sugar" count only inside a full match of '
        "that term."
    )
    example_cls = BWER_

    param_schema = KeyTermParams

    @metric_property(is_dependency=True)
    def _bu_wer_outcomes(self):
        """Get the shared _BUWEROutcomes metric instance."""
        return self.dataset.metrics._bu_wer_outcomes(
            vocab=self.params.vocab,
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_property(is_value=True)
    def num_biased_edits(self) -> int:
        """Total edits attributed to the biased (vocabulary) side."""
        return self._bu_wer_outcomes.num_biased_edits

    @metric_property(is_value=True)
    def num_biased_ref_tokens(self) -> int:
        """Total reference tokens attributed to the biased side."""
        return self._bu_wer_outcomes.num_biased_ref_tokens

    @metric_property(is_main=True)
    def value(self) -> float:
        """Biased word error rate (B-WER)."""
        if self.num_biased_ref_tokens == 0:
            return float(self.num_biased_edits)
        return self.num_biased_edits / self.num_biased_ref_tokens
