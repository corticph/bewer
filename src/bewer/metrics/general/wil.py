"""Word Information Lost (WIL) metric.

Defined by Morris, Maier and Green (2004), "From WER and RIL to MER and WIL: improved
evaluation measures for connected speech recognition", ICSLP, as 1 minus word
information preserved (WIP). Matches the jiwer reference implementation, which
computes WIL as exactly that difference.
"""

from __future__ import annotations

from bewer.metrics.base import METRIC_REGISTRY, ExampleMetric, Metric, metric_property

__all__ = ["WIL"]


class WIL_(ExampleMetric):
    @metric_property
    def _wip(self) -> float:
        """The example-level word information preserved this metric negates."""
        return self.parent_metric._wip[self.example.index].value

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the example-level word information lost."""
        return 1.0 - self._wip


@METRIC_REGISTRY.register("wil")
class WIL(Metric):
    short_name_base = "WIL"
    long_name_base = "Word Information Lost"
    description = (
        "Word information lost (WIL) is 1 minus word information preserved (WIP). It is 0 for a "
        "perfect transcription and 1 when no word information is communicated. It was defined by "
        "Morris, Maier and Green (2004, From WER and RIL to MER and WIL)."
    )
    example_cls = WIL_

    @metric_property(is_dependency=True)
    def _wip(self):
        """The shared WIP metric instance this metric negates."""
        return self.dataset.metrics.wip(
            standardizer=self.standardizer,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
        )

    @metric_property(is_main=True)
    def value(self) -> float:
        """Get the word information lost."""
        return 1.0 - self._wip.value
