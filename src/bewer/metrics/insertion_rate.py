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
        """Get the number of insertions in runs of length >= run_length."""
        if self.params.run_length == 1:
            return self._alignment.num_insertions
        return sum(length for length in self._insertion_runs() if length >= self.params.run_length)

    @metric_value
    def ref_length(self) -> int:
        """Get the number of tokens in the reference text."""
        if self.params.normalized:
            return len(self.example.ref.tokens.normalized)
        return len(self.example.ref.tokens.standardized)

    @metric_value(main=True)
    def value(self) -> float:
        """Get the example-level insertion rate."""
        if self.ref_length == 0:
            return float(self.num_insertions)
        return self.num_insertions / self.ref_length


@METRIC_REGISTRY.register("insertion_rate")
class InsertionRate(Metric):
    short_name_base = "IR"
    long_name_base = "Insertion Rate"
    description = (
        "Insertion rate (IR) is the number of insertions divided by the total number of "
        "tokens in the reference texts. It is useful for tracking hallucinations: a high "
        "insertion rate indicates the model is inventing text. The `run_length` parameter "
        "(default 1) controls the minimum contiguous insertion run length to count; setting "
        "it above 1 isolates burst insertions, which are a stronger hallucination signal."
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
    def ref_length(self) -> int:
        """Get the number of tokens in the reference texts."""
        return sum(em.ref_length for em in self)

    @metric_value(main=True)
    def value(self) -> float:
        """Get the insertion rate."""
        if self.ref_length == 0:
            return float(self.num_insertions)
        return self.num_insertions / self.ref_length
