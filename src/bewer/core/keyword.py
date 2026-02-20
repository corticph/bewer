from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from bewer.core.text import Text, TextType, TokenList
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.example import Example


class Keyword(Text):
    """A keyword that can locate itself within reference text tokens.

    Inherits standardized, tokens, and pipeline caching from Text.
    Adds contiguous token matching against a reference TokenList.
    """

    def __init__(
        self,
        raw: str,
        src: Optional["Example"] = None,
    ):
        super().__init__(raw=raw, src=src, text_type=TextType.KEYWORD)
        self._cache_find_in_ref: dict[tuple[str, str, str | None], list[TokenList]] = {}

    def find_in_tokens(
        self,
        token_list: TokenList,
        normalized: bool = True,
    ) -> list[TokenList]:
        """Find all contiguous occurrences of this keyword's tokens in the given TokenList.

        Args:
            token_list: The TokenList to search within (typically ref.tokens).
            normalized: Whether to match using normalized token text.

        Returns:
            list[TokenList]: Each element is a TokenList slice from token_list
                             representing one contiguous match.
        """
        kw_tokens = self.tokens.normalized if normalized else self.tokens.raw
        if len(kw_tokens) == 0:
            return []

        # Start with positions of the first keyword token
        starts = token_list.indices(kw_tokens[0], normalized=normalized)

        # Intersect with offset-shifted positions of each subsequent token
        for offset, kw_text in enumerate(kw_tokens[1:], start=1):
            positions = token_list.indices(kw_text, normalized=normalized)
            starts = starts & {p - offset for p in positions}
            if not starts:
                return []

        kw_len = len(kw_tokens)
        return [token_list[start : start + kw_len] for start in sorted(starts)]

    def find_in_ref(self, normalized: bool = True) -> list[TokenList]:
        """Find all contiguous occurrences of this keyword in the source example's reference tokens.

        Results are cached per active pipeline and normalized flag.

        Args:
            normalized: Whether to match using normalized token text.

        Returns:
            list[TokenList]: Matching token slices from the reference.

        Raises:
            ValueError: If source example is not set.
        """
        if self._src is None:
            raise ValueError("Source example is not set. Cannot search reference tokens.")
        cache_key = (
            STANDARDIZER_NAME.get(),
            TOKENIZER_NAME.get(),
            NORMALIZER_NAME.get() if normalized else None,
        )
        if cache_key not in self._cache_find_in_ref:
            self._cache_find_in_ref[cache_key] = self.find_in_tokens(self._src.ref.tokens, normalized=normalized)
        return self._cache_find_in_ref[cache_key]

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'Keyword("{text}")'
