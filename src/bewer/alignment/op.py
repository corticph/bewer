from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

from bewer.alignment.op_type import OpType
from bewer.style.html.alignment import generate_alignment_html
from bewer.style.html.color_schemes import HTMLDefaultAlignmentColors
from bewer.style.python.alignment import DefaultColorScheme, display_basic_aligned

if TYPE_CHECKING:
    from bewer.core.example import Example
    from bewer.style.html.color_schemes import HTMLAlignmentColors
    from bewer.style.python.alignment import ColorScheme


@dataclass
class Op:
    """Class representing an operation with its type.

    Attributes:
        type (OpType): The type of operation.
        ref (str | None): The reference string involved in the operation.
        hyp (str | None): The hypothesis string involved in the operation.
        ref_token_idx (int | None): The index of the token in the reference text (for word-level alignments).
        hyp_token_idx (int | None): The index of the token in the hypothesis text (for word-level alignments).
        ref_span (slice | None): The index span in the reference text.
        hyp_span (slice | None): The index span in the hypothesis text.
        hyp_left_partial (bool): Whether the hypothesis token is a partial word, cropped on the left.
        hyp_right_partial (bool): Whether the hypothesis token is a partial word, cropped on the right.
        ref_left_partial (bool): Whether the reference token is a partial word, cropped on the left.
        ref_right_partial (bool): Whether the reference token is a partial word, cropped on the right.
    """

    type: OpType
    ref: str | None = None
    hyp: str | None = None
    ref_token_idx: int | None = None
    hyp_token_idx: int | None = None
    ref_span: slice | None = None
    hyp_span: slice | None = None
    hyp_left_partial: bool = False
    hyp_right_partial: bool = False
    ref_left_partial: bool = False
    ref_right_partial: bool = False

    def __post_init__(self):
        if self.type == OpType.MATCH:
            if self.ref is None or self.hyp is None:
                raise ValueError("MATCH operation must have non-empty ref or hyp.")
        elif self.type == OpType.INSERT:
            if self.hyp is None or self.ref is not None:
                raise ValueError("INSERT operation must have non-empty hyp and empty ref.")
        elif self.type == OpType.DELETE:
            if self.hyp is not None or self.ref is None:
                raise ValueError("DELETE operation must have non-empty ref and empty hyp.")
        elif self.type == OpType.SUBSTITUTE:
            if self.ref is None or self.hyp is None:
                raise ValueError("SUBSTITUTE operation must have both ref and hyp.")

    @property
    def _repr_hyp(self) -> str:
        """Return the hypothesis with partial markers if applicable."""
        if self.hyp is None:
            return None
        return f'{"-" if self.hyp_left_partial else ""}"{self.hyp}"{"-" if self.hyp_right_partial else ""}'

    @property
    def _repr_ref(self) -> str:
        """Return the reference with partial markers if applicable."""
        if self.ref is None:
            return None
        return f'{"-" if self.ref_left_partial else ""}"{self.ref}"{"-" if self.ref_right_partial else ""}'

    def to_dict(self) -> dict:
        """Convert the Op instance to a dictionary."""
        return {
            "type": self.type.name,
            "ref": self.ref,
            "hyp": self.hyp,
            "ref_token_idx": self.ref_token_idx,
            "hyp_token_idx": self.hyp_token_idx,
            "ref_span": (self.ref_span.start, self.ref_span.stop) if self.ref_span else None,
            "hyp_span": (self.hyp_span.start, self.hyp_span.stop) if self.hyp_span else None,
            "hyp_left_partial": self.hyp_left_partial,
            "hyp_right_partial": self.hyp_right_partial,
            "ref_left_partial": self.ref_left_partial,
            "ref_right_partial": self.ref_right_partial,
        }

    def __repr__(self) -> str:
        if self.type == OpType.DELETE:
            return f"Op({self.type.name}: {self._repr_ref})"
        if self.type == OpType.INSERT:
            return f"Op({self.type.name}: {self._repr_hyp})"
        if self.type == OpType.SUBSTITUTE:
            return f"Op({self.type.name}: {self._repr_hyp} -> {self._repr_ref})"
        return f"Op({self.type.name}: {self._repr_hyp} == {self._repr_ref})"


class Alignment(list[Op]):
    """
    List of operations with additional methods for processing.

    Attributes:
        ops (list[Op]): List of operations.
    """

    _src_example: "Example" | None = None

    def set_source(self, src: "Example"):
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

    def to_html(
        self,
        max_line_length: int = 200,
        color_scheme: type[HTMLAlignmentColors] = HTMLDefaultAlignmentColors,
    ) -> str:
        """Render the alignment as an HTML string.

        Args:
            max_line_length (int): Maximum line length for wrapping. Defaults to 100.
            color_scheme (type[HTMLAlignmentColors]): Color scheme for display.

        Returns:
            str: HTML string representing the alignment visualization.
        """
        title = None if self._src_example is None else f"   Example {self._src_example.index}"
        return generate_alignment_html(
            self,
            max_line_length=max_line_length,
            title=title,
            color_scheme=color_scheme,
        )

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
        return Alignment(super().__add__(other))

    def __repr__(self):
        ops = self[:60]
        ops_str = ",\n ".join([repr(op) for op in ops])
        if len(self) > 60:
            ops_str += ",\n ..."
        return f"Alignment([\n {ops_str}]\n)"
