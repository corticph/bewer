from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import MetricParams

__all__ = ["InsertionParams"]


@dataclass
class InsertionParams(MetricParams):
    """Base class for insertion metric parameter dataclasses.

    Provides the common ``min_run_length``, ``normalized``, and ``alignment``
    fields and validates them.

    Attributes:
        min_run_length: Minimum contiguous insertion run length to count.
        normalized: Whether to use normalized tokens for alignment.
        alignment: Alignment backend — 'levenshtein' or 'error_align'.
    """

    min_run_length: int = 1
    normalized: bool = True
    alignment: str = "levenshtein"

    def validate(self) -> None:
        """Validate insertion parameters."""
        if self.min_run_length < 1:
            raise ValueError(f"min_run_length must be >= 1, got {self.min_run_length}.")
        if self.alignment not in ("levenshtein", "error_align"):
            raise ValueError(f"alignment must be 'levenshtein' or 'error_align', got '{self.alignment}'.")
