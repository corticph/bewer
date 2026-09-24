from __future__ import annotations

from dataclasses import dataclass

from bewer.alignment import Alignment
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value

__all__ = ["InsertionRunRate"]


class InsertionRunRate_(ExampleMetric):
    @property
    def _alignment(self) -> Alignment:
        return self.parent_metric._levenshtein[self.example.index].alignment

    @metric_value
    def num_insertions(self) -> int:
        """Get the total number of insertions in the example."""
        return self._alignment.num_insertions

    @metric_value
    def num_run_insertions(self) -> int:
        """Get the number of insertions that are part of contiguous runs of length >= min_run_length."""
        return sum(length for length in self._alignment.insertion_runs if length >= self.params.min_run_length)

    @metric_value
    def num_ops(self) -> int:
        """Get the total number of operations (edits + matches) in the example."""
        return self._alignment.num_ops

    @metric_value
    def max_run_length(self) -> int:
        """Get the length of the longest contiguous insertion run in the example."""
        return self._alignment.max_insertion_run_length

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level insertion run rate."""
        if self.num_ops == 0:
            return 0.0
        return self.num_run_insertions / self.num_ops


@METRIC_REGISTRY.register("insertion_run_rate")
class InsertionRunRate(Metric):
    short_name_base = "IRR"
    long_name_base = "Insertion Run Rate"
    description = (
        "Insertion run rate (IRR) is computed as the number of insertions that are part of "
        "contiguous insertion runs of length at least `min_run_length`, divided by the total "
        "number of operations (edits + matches). "
        "When min_run_length=1 (default), all insertions are counted, making the metric "
        "equivalent to the insertion rate. "
        "When min_run_length>=2, only insertions that are part of burst insertion runs are "
        "counted, which serves as a hallucination signal: a burst of consecutive inserted "
        "tokens is the signature of a model inventing text, as opposed to scattered "
        "single-token insertions."
    )
    example_cls = InsertionRunRate_

    @dataclass
    class param_schema(MetricParams):
        min_run_length: int = 1
        normalized: bool = True

        def validate(self) -> None:
            if self.min_run_length < 1:
                raise ValueError(f"min_run_length must be >= 1, got {self.min_run_length}.")

    @dependency
    def _levenshtein(self):
        return self.dataset.metrics.levenshtein(
            normalized=self.params.normalized,
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_value
    def num_insertions(self) -> int:
        """Get the total number of insertions across all examples."""
        return sum(em.num_insertions for em in self)

    @metric_value
    def num_run_insertions(self) -> int:
        """Get the total number of insertions in qualifying runs across all examples."""
        return sum(em.num_run_insertions for em in self)

    @metric_value
    def num_ops(self) -> int:
        """Get the total number of operations (edits + matches) across all examples."""
        return sum(em.num_ops for em in self)

    @metric_value(main=True)
    def value(self) -> float:
        """Get the insertion run rate."""
        if self.num_ops == 0:
            return 0.0
        return self.num_run_insertions / self.num_ops
