from typing import Optional

from bewer.core.keyword import Keyword, KeywordTrie
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME


def get_keyword_trie(
    vocabs: dict[str, set[Keyword]],
    cache: dict[tuple, Optional[KeywordTrie]],
    vocab: str,
    normalized: bool = True,
    add_capitalized: bool = False,
) -> Optional["KeywordTrie"]:
    """Get or build a trie for the keywords in the specified vocabulary."""
    trie_key = (
        STANDARDIZER_NAME.get(),
        TOKENIZER_NAME.get(),
        NORMALIZER_NAME.get() if normalized else None,
        add_capitalized,
        vocab,
    )
    if trie_key in cache:
        return cache[trie_key]

    keywords = vocabs.get(vocab, None)
    if keywords is None:
        return None

    trie = KeywordTrie(
        keywords,
        normalized=normalized,
        add_capitalized=add_capitalized,
    )
    cache[trie_key] = trie
    return trie
