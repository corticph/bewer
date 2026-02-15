from functools import cached_property
from typing import TYPE_CHECKING, Optional

import regex as re

from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME
from bewer.reporting.python.utils import highlight_span

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
        src: Optional["Text"] = None,
    ):
        """Initialize Token.

        Args:
            raw: The raw token string.
            start: Starting character index in the source text.
            end: Ending character index in the source text.
            index: Token index in the token list.
            src: Parent Text object. Can be set later via set_source().
        """
        self.raw = raw
        self.start = start
        self.end = end
        self.index = index
        self.slice = slice(self.start, self.end)

        self._normalized = {}

        self._src = None
        self._pipelines = None
        if src is not None:
            self.set_source(src)

    @property
    def src(self) -> Optional["Text"]:
        """Get the parent Text object."""
        return self._src

    def set_source(self, src: "Text") -> None:
        """Set the parent Text object.

        Args:
            src: The parent Text object.

        Raises:
            ValueError: If source is already set.
        """
        if self._src is not None:
            raise ValueError("Source already set for Token")

        self._src = src

        # Cache derived references for performance
        _src_example = src.src if src is not None else None
        _src_dataset = _src_example.src if _src_example is not None else None
        self._pipelines = _src_dataset.pipelines if _src_dataset is not None else None

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
        if self._src is None:
            return None
        _src_example = self._src.src
        if _src_example is None:
            return None
        return _src_example.levenshtein._token_to_op_index.get(self, None)

    def inctx(self, width: int = 20, highlight: bool = False, add_ellipsis: bool = True) -> str:
        """Get the context of the token in the source text.

        Args:
            width (int): The width of the context to show.
            add_ellipsis (bool): Whether to add ellipsis around the context, if starting/ending not within width.

        Returns:
            str: The context string.
        """
        if self._src is None:
            raise ValueError("Source text is not set. Cannot get context.")
        start = max(0, self.start - width)
        end = min(len(self._src.raw), self.end + width)
        ctx_span = self._src.raw[start:end]
        if highlight:
            ctx_span = highlight_span(ctx_span, self.start - start, self.end - start, "bold green")
        if add_ellipsis:
            start_marker = "..." if self.start - width > 0 else ""
            end_marker = "..." if self.end + width < len(self._src.raw) else ""
            ctx_span = start_marker + ctx_span + end_marker
        return ctx_span

    @classmethod
    def from_match(
        cls,
        match: re.Match,
        index: int,
        src: Optional["Text"] = None,
    ) -> "Token":
        """
        Create a Token object from a regex match object.

        Args:
            match (re.Match): The regex match object.
            index (int): Token index in the token list.
            src (Text): Parent Text object.

        Returns:
            Token: The created Token object.
        """
        return cls(
            raw=match.group(),
            start=match.start(),
            end=match.end(),
            index=index,
            src=src,
        )

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.start == other.start and self.end == other.end and self.raw == other.raw

    def __repr__(self):
        return f'Token("{self.raw}")'
