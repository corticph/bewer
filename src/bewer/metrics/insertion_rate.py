from __future__ import annotations

from dataclasses import dataclass

from bewer.alignment import Alignment
from bewer.alignment.op_type import OpType
from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, MetricParams, dependency, metric_value

__all__ = ["InsertionRate"]


class InsertionRate_(ExampleMetric):
    @property
    def _alignment(self) -> Alignment:
        return self.parent_metric._levenshtein[self.example.index].alignment

    def _insertion_runs(self) -> list[int]:
        """Get the lengths of all contiguous insertion runs in the example alignment."""
        runs = []
        current = 0
        for op in self._alignment:
            if op.type == OpType.INSERT:
                current += 1
            elif current > 0:
                runs.append(current)
                current = 0
        if current > 0:
            runs.append(current)
        return runs

    @metric_value
    def num_insertions(self) -> int:
        """Get the total number of insertions in the example."""
        return self._alignment.num_insertions

    @metric_value
    def num_run_insertions(self) -> int:
        """Get the number of insertions in contiguous runs of length >= run_length."""
        return sum(length for length in self._insertion_runs() if length >= self.params.run_length)

    @metric_value
    def num_ops(self) -> int:
        """Get the total number of operations (edits + matches) in the example."""
        return self._alignment.num_ops

    @metric_value
    def max_run_length(self) -> int:
        """Get the length of the longest contiguous insertion run in the example."""
        runs = self._insertion_runs()
        return max(runs) if runs else 0

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level insertion rate."""
        if self.num_ops == 0:
            return 0.0
        return self.num_run_insertions / self.num_ops


@METRIC_REGISTRY.register("insertion_rate")
class InsertionRate(Metric):
    short_name_base = "IR"
    long_name_base = "Insertion Rate"
    description = (
        "Insertion rate (IR) is computed as the number of insertions that are part of "
        "contiguous insertion runs of length at least `run_length`, divided by the total "
        "number of operations (edits + matches). "
        "When run_length=1 (default), all insertions are counted. "
        "When run_length>=2, only insertions that are part of burst insertion runs are "
        "counted, which serves as a hallucination signal: a burst of consecutive inserted "
        "tokens is the signature of a model inventing text, as opposed to scattered "
        "single-token insertions."
    )
    example_cls = InsertionRate_

    @dataclass
    class param_schema(MetricParams):
        run_length: int = 1
        normalized: bool = True

        def validate(self) -> None:
            if self.run_length < 1:
                raise ValueError(f"run_length must be >= 1, got {self.run_length}.")

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
        """Get the insertion rate."""
        if self.num_ops == 0:
            return 0.0
        return self.num_run_insertions / self.num_ops
