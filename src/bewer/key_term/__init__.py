"""Key term matching, vocabulary definitions, and extractors.

This package groups the key-term machinery:

- :class:`KeyTerm` and :class:`KeyTermMatch` — the data containers for key terms and matches.
- :class:`KeyTermTrie` — the Aho-Corasick matching engine.
- :class:`Vocabulary` — a named collection of key terms (static lists, files, extractors).
- :class:`RegexExtractor` and :class:`OrthographicallyComplexTermExtractor` — pre-built
  extractor functions that derive key terms from a dataset's references.
"""

from bewer.key_term.key_term import KeyTerm, KeyTermMatch
from bewer.key_term.orthographically_complex_term_extractor import (
    ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN,
    OrthographicallyComplexTermExtractor,
)
from bewer.key_term.regex_extractor import RegexExtractor, match_token_regex
from bewer.key_term.trie import KeyTermTrie
from bewer.key_term.vocabulary import (
    ExtractorFn,
    Vocabulary,
    VocabularyExtractorError,
    VocabularyFrozenError,
)

__all__ = [
    "KeyTerm",
    "KeyTermMatch",
    "KeyTermTrie",
    "Vocabulary",
    "VocabularyExtractorError",
    "VocabularyFrozenError",
    "ExtractorFn",
    "RegexExtractor",
    "match_token_regex",
    "OrthographicallyComplexTermExtractor",
    "ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN",
]
