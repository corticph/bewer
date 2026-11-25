from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bewer.metrics.base import MetricParams

__all__ = ["InsertionParams"]


@dataclass
class InsertionParams(MetricParams):
    """Base class for insertion metric parameter dataclasses.

    Provides the common ``min_run_length`` and ``alignment``
    fields and validates them.

    Attributes:
        min_run_length: Minimum contiguous insertion run length to count.
        alignment: Registry name of the alignment metric to use, 'levenshtein' or 'error_align'.
            The ``Literal`` type lists the allowed values, which also lets the
            ``selected_by`` dependency on it know its possible targets.
    """

    min_run_length: int = 1
    alignment: Literal["levenshtein", "error_align"] = "levenshtein"

    def validate(self) -> None:
        """Validate insertion parameters."""
        if self.min_run_length < 1:
            raise ValueError(f"min_run_length must be >= 1, got {self.min_run_length}.")
