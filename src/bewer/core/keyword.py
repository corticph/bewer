from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from bewer.core.text import Text, TextType, TokenList

if TYPE_CHECKING:
    from bewer.core.example import Example

__all__ = ["Keyword"]


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

    def __repr__(self):
        text = self.raw if len(self.raw) <= 46 else self.raw[:46] + "..."
        return f'Keyword("{text}")'


class KeywordTrie:
    """A trie for efficient keyword matching in reference tokens."""

    def __init__(
        self,
        keywords: set[Keyword],
        normalized: bool = True,
        add_capitalized: bool = False,
    ):
        """Initialize the trie with the given set of keywords.

        Args:
            keywords: A set of Keyword objects to build the trie from.
            normalized: Whether to use normalized text for matching. If False, uses raw text.
            add_capitalized: Whether to add capitalized versions of keywords to the trie for case-insensitive matching.
                Only applies if normalized is False and is only applied to the first word in an n-gram keyword.
        """
        # TODO: Consider whether to add capitalized versions of all tokens in the keyword phrase (not just the first).
        # TODO: Consider whether to add capitalized versions of keywords when normalized is True.
        self.children = {}
        self.normalized = normalized
        self.add_capitalized = add_capitalized
        self.build(keywords)
        self.root_tokens = frozenset(self.children.keys())

    def build(self, keywords: set[Keyword]) -> None:
        """Build the trie from the given set of keywords."""
        apply_capitalization = self.add_capitalized and not self.normalized

        for keyword in keywords:
            tokens = keyword.tokens.normalized if self.normalized else keyword.tokens.raw
            self.add_path(tokens)
            if apply_capitalization:
                first_capitalized = tokens[0].capitalize()
                if first_capitalized != tokens[0]:
                    tokens[0] = first_capitalized
                    self.add_path(tokens)

    def add_path(self, tokens: list[str]) -> None:
        """Add a path of tokens to the trie."""
        current_node = self
        for token in tokens:
            if token not in current_node.children:
                current_node.children[token] = KeywordNode()
            current_node = current_node.children[token]
        current_node.is_end = True

    def find_in_tokens(self, tokens: TokenList, allow_subsets: bool = True) -> list[slice]:
        """Find all contiguous token sequences in the given tokens that match any keyword in the trie."""
        # Early exit: skip scan if no root tokens appear in the text
        tokens = tokens.normalized if self.normalized else tokens.raw
        if self.root_tokens.isdisjoint(set(tokens)):
            return []

        matches = []
        for i in range(len(tokens)):
            current_node = self
            for j in range(i, len(tokens)):
                token_text = tokens[j]
                if token_text not in current_node.children:
                    break
                current_node = current_node.children[token_text]
                if current_node.is_end:
                    if not allow_subsets and matches:
                        if matches[-1].start == i:
                            matches[-1] = slice(matches[-1].start, j + 1)
                        elif matches[-1].stop >= j + 1:
                            continue
                    else:
                        matches.append(slice(i, j + 1))
        return matches


class KeywordNode:
    """A node in the keyword trie, representing a single token in a keyword."""

    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}
        self.is_end = False
