from typing import TYPE_CHECKING, Optional

import regex as re

from bewer.core.caching import pipeline_cached_property
from bewer.preprocessing.context import NORMALIZER_NAME
from bewer.reporting.python.utils import highlight_span

if TYPE_CHECKING:
    from bewer.core.text import Text

__all__ = ["Token"]


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
        *,
        src: "Text",
    ):
        """Initialize Token.

        Args:
            raw: The raw token string.
            start: Starting character index in the source text.
            end: Ending character index in the source text.
            index: Token index in the token list.
            src: Parent Text object (required).
        """
        self._raw = raw
        self.start = start
        self.end = end
        self.index = index
        self.slice = slice(self.start, self.end)

        self._cache_normalized = {}

        self._src = src
        self._pipelines = src.pipelines

    @property
    def src(self) -> "Text":
        """Get the parent Text object."""
        return self._src

    @property
    def raw(self) -> str:
        """The raw string of the token as extracted during tokenization."""
        return self._raw

    @pipeline_cached_property(NORMALIZER_NAME)
    def normalized(self, normalizer):
        """The normalized string of the token after applying the active normalizer."""
        return normalizer(self.raw)

    def inctx(self, width: int = 20, highlight: bool = False, add_ellipsis: bool = True) -> str:
        """Get the context of the token in the source text.

        Args:
            width (int): The number of characters of context to show on each side.
            highlight (bool): Whether to highlight the token span in the context.
            add_ellipsis (bool): Whether to add ellipsis around the context, if starting/ending not within width.

        Returns:
            str: The context string.
        """
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
        src: "Text",
    ) -> "Token":
        """
        Create a Token object from a regex match object.

        Args:
            match (re.Match): The regex match object.
            index (int): Token index in the token list.
            src (Text): Parent Text object (required).

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
