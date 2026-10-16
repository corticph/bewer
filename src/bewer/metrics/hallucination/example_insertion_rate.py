from __future__ import annotations

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_property
from bewer.metrics.hallucination._insertion_params import InsertionParams

__all__ = ["ExampleInsertionRate"]


class ExampleInsertionRate_(ExampleMetric):
    @metric_property(is_value=True)
    def num_insertions(self) -> int:
        """Get the number of qualifying insertions in this example."""
        return self.parent_metric._insertion_rate[self.example.index].num_insertions

    @metric_property(is_value=True)
    def has_insertions(self) -> bool:
        """Whether the example has any qualifying insertions."""
        return self.num_insertions > 0


@METRIC_REGISTRY.register("example_insertion_rate")
class ExampleInsertionRate(Metric):
    short_name_base = "EIR"
    long_name_base = "Example Insertion Rate"
    description = (
        "Example Insertion Rate (EIR) is the fraction of examples that have at least one "
        "insertion (filtered by min_run_length). Unlike Insertion Rate, which counts "
        "insertions per token, EIR counts the proportion of affected examples. This is "
        "useful because hallucinations often stem from unrecoverable autoregressive/attention "
        "collapse (i.e., a failure mode that affects the model once), making the presence of "
        "insertions effectively a per-example binary signal. "
        "The `min_run_length` parameter (default 1) controls the minimum contiguous "
        "insertion run length to count; setting it above 1 isolates burst insertions. "
        "The `alignment` parameter (default 'levenshtein') selects the alignment backend: "
        "'levenshtein' (word-level via RapidFuzz) or 'error_align' (via the error-align "
        "package)."
    )
    example_cls = ExampleInsertionRate_

    param_schema = InsertionParams

    @metric_property(is_dependency=True)
    def _insertion_rate(self):
        return self.dataset.metrics.insertion_rate(
            min_run_length=self.params.min_run_length,
            normalized=self.params.normalized,
            alignment=self.params.alignment,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_property(is_value=True)
    def num_examples_with_insertions(self) -> int:
        """Get the number of examples with at least one qualifying insertion."""
        return sum(1 for em in self if em.has_insertions)

    @metric_property(is_value=True)
    def num_examples(self) -> int:
        """Get the total number of examples."""
        return len(self)

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the fraction of examples with at least one qualifying insertion."""
        if self.num_examples == 0:
            return 0.0
        return self.num_examples_with_insertions / self.num_examples
