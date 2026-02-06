from functools import cached_property
from typing import TYPE_CHECKING, Optional

import regex as re

from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME
from bewer.style.python.utils import highlight_span

if TYPE_CHECKING:
    from bewer.alignment.op import Op
    from bewer.core.text import Text


class Token:
    """BeWER Token representation.

    Attributes:
        raw (str): The raw string of the token.
        start (int): The starting index of the token in the text.
        end (int): The ending index of the token in the text.
        index (int | None): The index of the token in the token list.
        slice (slice): A slice object representing the token's position in the text.
        normalized (str | None): The normalized string of the token.
    """

    def __init__(
        self,
        raw: str,
        start: int,
        end: int,
        index: Optional[int] = None,
        _src_text: Optional["Text"] = None,
    ):
        self.raw = raw
        self.start = start
        self.end = end
        self.index = index
        self.slice = slice(self.start, self.end)

        self._normalized = {}

        self._src_text = _src_text
        self._src_example = _src_text._src_example if _src_text else None
        self._src_dataset = self._src_example._src_dataset if self._src_example else None
        self._pipelines = self._src_dataset.pipelines if self._src_dataset else None

    @property
    def normalized(self) -> str:
        """Get the normalized string of the token using a specified normalizer.

        Returns:
            str: The normalized string.
        """
        standardizer_name = STANDARDIZER_NAME.get()
        tokenizer_name = TOKENIZER_NAME.get()
        normalizer_name = NORMALIZER_NAME.get()
        pipeline_key = (standardizer_name, tokenizer_name, normalizer_name)

        if pipeline_key in self._normalized:
            return self._normalized[pipeline_key]
        if self._pipelines is None:
            raise ValueError("No normalizers found in pipelines.")
        normalizer_func = self._pipelines.normalizers.get(normalizer_name, None)
        if normalizer_func is None:
            raise ValueError(f"Token normalizer '{normalizer_name}' not found in pipelines.")
        normalized_token = normalizer_func(self.raw)
        self._normalized[pipeline_key] = normalized_token
        return normalized_token

    @cached_property
    def levenshtein(self) -> "Op":
        if self._src_example is None:
            return None
        return self._src_example.levenshtein._token_to_op_index.get(self, None)

    def inctx(self, width: int = 20, highlight: bool = False, add_ellipsis: bool = True) -> str:
        """Get the context of the token in the source text.

        Args:
            width (int): The width of the context to show.
            add_ellipsis (bool): Whether to add ellipsis around the context, if starting/ending not within width.

        Returns:
            str: The context string.
        """
        if self._src_text is None:
            raise ValueError("Source text is not set. Cannot get context.")
        start = max(0, self.start - width)
        end = min(len(self._src_text.raw), self.end + width)
        ctx_span = self._src_text.raw[start:end]
        if highlight:
            ctx_span = highlight_span(ctx_span, self.start - start, self.end - start, "bold green")
        if add_ellipsis:
            start_marker = "..." if self.start - width > 0 else ""
            end_marker = "..." if self.end + width < len(self._src_text.raw) else ""
            ctx_span = start_marker + ctx_span + end_marker
        return ctx_span

    @classmethod
    def from_match(
        cls,
        match: re.Match,
        index: int,
        _src_text: Optional["Text"] = None,
    ) -> "Token":
        """
        Create a Token object from a regex match object.

        Args:
            match (re.Match): The regex match object.

        Returns:
            Token: The created Token object.
        """
        return cls(
            raw=match.group(),
            start=match.start(),
            end=match.end(),
            index=index,
            _src_text=_src_text,
        )

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.start == other.start and self.end == other.end and self.raw == other.raw

    def __repr__(self):
        return f'Token("{self.raw}")'
