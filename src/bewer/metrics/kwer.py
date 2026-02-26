from __future__ import annotations

from dataclasses import dataclass

from bewer.alignment import OpType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, metric_value

__all__ = ["KWER", "KWER_"]


class KWER_(ExampleMetric):
    def _get_alignment(self):
        return self.example.metrics.levenshtein(
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        ).alignment

    @metric_value
    def num_errors(self) -> int:
        """Get the number of keywords incorrectly transcribed in the hypothesis text."""
        keywords = self.example.keywords.get(self.params.vocab, None)
        if keywords is None:
            return 0
        alignment = self._get_alignment()
        num_errors = 0
        for keyword in keywords:
            ref_token_lists = keyword.find_in_ref(normalized=True)
            if not ref_token_lists:
                continue
            for tokens in ref_token_lists:
                if len(tokens) == 1:
                    ops = alignment.ops_from_ref_index(tokens[0].index)
                else:
                    ops = alignment.ops_from_ref_index(tokens[0].index, tokens[-1].index)
                if any(op.type != OpType.MATCH for op in ops):
                    num_errors += 1
        return num_errors

    @metric_value
    def num_keywords(self) -> int:
        """Get the number of keywords in the reference text."""
        keywords = self.example.keywords.get(self.params.vocab, None)
        if keywords is None:
            return 0
        return sum(len(keyword.find_in_ref(normalized=True)) for keyword in keywords)

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level keyword error rate."""
        if self.num_keywords == 0:
            return float(self.num_errors)
        return self.num_errors / self.num_keywords


@METRIC_REGISTRY.register("kwer")
class KWER(Metric):
    short_name_base = "KWER"
    long_name_base = "Keyword Error Rate"
    description = (
        "Keyword error rate (KWER) is computed as the number of keywords (or key terms) incorrectly transcribed in "
        "the hypothesis texts, divided by the total number of keywords identified in the reference texts. A keyword "
        "may consist of one or more tokens, but is treated as a single unit for the purpose of KWER calculation."
    )
    example_cls = KWER_

    @dataclass
    class param_schema(MetricParams):
        """Parameters for the KWER metric.

        Attributes:
            vocab: The vocabulary name to use for keyword identification.
            normalized: Whether to use normalized tokens for alignment and keyword matching.
        """

        vocab: str
        normalized: bool = True

        def validate(self) -> None:
            """Validate that the metric can be computed with the given parameters and source data."""
            is_dynamic_vocab = self.vocab in self.metric.dataset._dynamic_keyword_vocabs
            is_static_vocab = self.vocab in self.metric.dataset._static_keyword_vocabs
            if not is_dynamic_vocab and not is_static_vocab:
                raise ValueError(f"Vocabulary '{self.vocab}' not found in dataset keyword vocabularies.")

    @metric_value
    def num_errors(self) -> int:
        """Get the number of keywords incorrectly transcribed in the hypothesis texts."""
        return sum([self.get_example_metric(example).num_errors for example in self._src])

    @metric_value
    def num_keywords(self) -> int:
        """Get the number of keywords in the reference texts."""
        return sum([self.get_example_metric(example).num_keywords for example in self._src])

    @metric_value(main=True)
    def value(self) -> float:
        """Get the keyword error rate."""
        if self.num_keywords == 0:
            return float(self.num_errors)
        return self.num_errors / self.num_keywords
