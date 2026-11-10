from __future__ import annotations

from dataclasses import dataclass

from bewer.metrics.base import AlignmentParams

__all__ = ["InsertionParams"]


@dataclass
class InsertionParams(AlignmentParams):
    """Base class for insertion metric parameter dataclasses.

    Provides the common ``min_run_length`` field on top of the ``alignment`` field from
    :class:`AlignmentParams`, and validates both.

    Attributes:
        min_run_length: Minimum contiguous insertion run length to count.
    """

    min_run_length: int = 1

    def validate(self) -> None:
        """Validate insertion parameters."""
        super().validate()
        if self.min_run_length < 1:
            raise ValueError(f"min_run_length must be >= 1, got {self.min_run_length}.")
