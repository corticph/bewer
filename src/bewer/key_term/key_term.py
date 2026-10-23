from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Optional

from bewer.core.text import Text, TextType, TokenList

if TYPE_CHECKING:
    from bewer.configs.config import BewerConfig

__all__ = ["KeyTerm", "KeyTermMatch"]


class KeyTerm(Text):
    """A key term that can locate itself within reference text tokens.

    Inherits standardized, tokens, and pipeline caching from Text.
    Adds contiguous token matching against a reference TokenList.
    """

    def __init__(
        self,
        raw: str,
        *,
        config: "BewerConfig",
    ):
        super().__init__(raw=raw, config=config, src=None, text_type=TextType.KEY_TERM)

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'KeyTerm("{text}")'


class KeyTermMatch:
    """A structured key term match: a token span in a ``Text`` identified as a ``KeyTerm``.

    Carries the matched token span (``start``/``stop`` token indices, mirroring the raw
    ``slice`` previously returned so existing consumers that read ``.start``/``.stop`` keep
    working) together with the source ``Text`` it was found in (``src``, mirroring
    ``Token.src``) and the ``key_term`` it was identified as. Richer details — the side it
    was found on and the matched tokens — are derived from those two references.
    """

    def __init__(
        self,
        start: int,
        stop: int,
        src: "Text",
        key_term: "KeyTerm",
    ):
        self.start = start
        self.stop = stop
        self.src = src
        self.key_term = key_term

    @cached_property
    def token_slice(self) -> slice:
        """The matched span as a ``slice`` for indexing the source text's ``TokenList``."""
        return slice(self.start, self.stop)

    @property
    def side(self) -> Optional[TextType]:
        """Which side the match was found on (``REF`` / ``HYP`` / ``KEY_TERM``), or ``None``
        if the source text has no type set."""
        return self.src.text_type

    @cached_property
    def tokens(self) -> TokenList:
        """The matched ``Token`` objects from the source text."""
        return self.src.tokens[self.token_slice]

    def __repr__(self):
        return f"KeyTermMatch(term={self.key_term.raw!r}, span=({self.start}, {self.stop}))"
