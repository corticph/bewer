from __future__ import annotations

import json
from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

from bewer.alignment.op_type import OpType
from bewer.reporting.html.alignment import generate_alignment_html_lines as _generate_alignment_html_lines
from bewer.reporting.html.color_schemes import HTMLDefaultAlignmentColors
from bewer.reporting.python.alignment import DefaultColorScheme, display_basic_aligned

if TYPE_CHECKING:
    from bewer.alignment.op import Op
    from bewer.core.example import Example
    from bewer.reporting.html.color_schemes import HTMLAlignmentColors
    from bewer.reporting.python.alignment import ColorScheme


class Alignment(list["Op"]):
    """
    List of operations with additional methods for processing.

    Attributes:
        ops (list[Op]): List of operations.
        num_matches (int): Number of match operations.
        num_substitutions (int): Number of substitution operations.
        num_insertions (int): Number of insertion operations.
        num_deletions (int): Number of deletion operations.
    """

    def __init__(self, iterable: Iterable["Op"] = (), src_example: Optional["Example"] = None) -> None:
        super().__init__(iterable)
        self._op_counts = Counter()
        self._count_operations(self)
        self._src_example = src_example

        for op in self:
            op.set_source(self)

    @property
    def num_matches(self) -> int:
        """Get the number of match operations."""
        return self._op_counts[OpType.MATCH]

    @property
    def num_substitutions(self) -> int:
        """Get the number of substitution operations."""
        return self._op_counts[OpType.SUBSTITUTE]

    @property
    def num_insertions(self) -> int:
        """Get the number of insertion operations."""
        return self._op_counts[OpType.INSERT]

    @property
    def num_deletions(self) -> int:
        """Get the number of deletion operations."""
        return self._op_counts[OpType.DELETE]

    @property
    def num_edits(self) -> int:
        """Get the total number of edit operations (substitutions, insertions, deletions)."""
        return self.num_substitutions + self.num_insertions + self.num_deletions

    @cached_property
    def _start_index_mapping(self) -> dict[int, int]:
        """Create a mapping from character start index to token index for quick lookup."""
        mapping = {}
        for i, op in enumerate(self):
            if op.ref_span is not None:
                mapping[op.ref_span.start] = i
        return mapping

    @cached_property
    def _end_index_mapping(self) -> dict[int, int]:
        """Create a mapping from character end index to token index for quick lookup."""
        mapping = {}
        for i, op in enumerate(self):
            if op.ref_span is not None:
                mapping[op.ref_span.stop] = i
        return mapping

    def start_index_to_op(self, char_index: int) -> Optional[Op]:
        """Get the op that starts at the given character index.

        Args:
            char_index (int): The character index to look up.

        Returns:
            Optional[Op]: The op that starts at the given character index, or None if not found.
        """
        op_index = self._start_index_mapping.get(char_index, None)
        if op_index is not None:
            return self[op_index]
        return None

    def end_index_to_op(self, char_index: int) -> Optional[Op]:
        """Get the op that ends at the given character index.

        Args:
            char_index (int): The character index to look up.

        Returns:
            Optional[Op]: The op that ends at the given character index, or None if not found.
        """
        op_index = self._end_index_mapping.get(char_index, None)
        if op_index is not None:
            return self[op_index]
        return None

    def append(self, op: "Op") -> None:
        if self._src_example is not None:
            raise ValueError("Cannot modify Alignment after source example is set.")
        super().append(op)
        self._count_operations([op])

    def extend(self, iterable: Iterable["Op"]) -> None:
        if self._src_example is not None:
            raise ValueError("Cannot modify Alignment after source example is set.")
        ops = list(iterable)
        super().extend(ops)
        self._count_operations(ops)

    def _count_operations(self, ops: Iterable["Op"]) -> None:
        """Count operation types and store as attributes."""
        for op in ops:
            self._op_counts[op.type] += 1

    def set_source(self, src: "Example") -> None:
        """Set the source example for the alignment."""
        self._src_example = src

    def to_dicts(self) -> list[dict]:
        """Dump the alignment to a list of dictionaries.

        Returns:
            list[dict]: List of operations as dictionaries.
        """
        return [op.to_dict() for op in self]

    def to_json(self, path: str | None = None, allow_overwrite: bool = False) -> str:
        """Dump the alignment to a JSON string.

        Args:
            path (str | None): If provided, write the JSON string to this file.
            allow_overwrite (bool): If True, overwrite the file if it exists.
        Returns:
            str: JSON string representing the alignment.
        """
        json_str = json.dumps(self.to_dicts(), indent=2)
        if path is not None:
            path = Path(path)
            if path.is_dir():
                raise ValueError("Provided path is a directory, expected a file path.")
            if path.exists() and not allow_overwrite:
                raise FileExistsError(f"File {path} already exists.")
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(json_str)
        return json_str

    def display(
        self,
        max_line_length: int | float = 0.5,
        color_scheme: ColorScheme = DefaultColorScheme,
    ) -> None:
        """Display the alignment in the console.

        Args:
            max_line_length (int | None): Maximum line length for display. If None, uses default.
            color_scheme (ColorScheme): Color scheme for display.
        """
        title = None if self._src_example is None else f"   Example {self._src_example.index}"
        display_basic_aligned(self, max_line_length=max_line_length, title=title, color_scheme=color_scheme)

    def _to_html_lines(
        self,
        color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    ) -> list[tuple[str, str]]:
        """Render the alignment as an HTML string.

        Args:
            color_scheme (type[HTMLAlignmentColors]): Color scheme for display.

        Returns:
            list[tuple[str, str]]: List of tuples containing HTML strings representing the alignment visualization.
        """
        return _generate_alignment_html_lines(self, color_scheme=color_scheme)

    def __getitem__(self, index: int) -> Union[Op, "Alignment"]:
        if isinstance(index, slice):
            return Alignment(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: "Alignment") -> "Alignment":
        """Concatenate two Alignment objects.

        Args:
            other (Alignment): The other Alignment object.
        Returns:
            Alignment: The concatenated Alignment object.
        """
        if self._src_example is not None or other._src_example is not None:
            raise ValueError("Cannot concatenate Alignments with source examples set.")
        return Alignment(super().__add__(other))

    def __repr__(self):
        ops = self[:60]
        ops_str = ",\n ".join([repr(op) for op in ops])
        if len(self) > 60:
            ops_str += ",\n ..."
        return f"Alignment([\n {ops_str}]\n)"
