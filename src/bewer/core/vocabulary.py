from __future__ import annotations

import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional

from bewer.core.key_term import (
    KeyTerm,
    KeyTermTrie,
    _remove_duplicate_matches,
    _remove_subset_matches,
)
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.text import Text

__all__ = ["Vocabulary", "VocabularyExtractorError", "ExtractorFn"]

# An extractor takes a Dataset and returns key term strings pulled from its examples
# (reference or hypothesis text). Extracted terms are treated exactly like static terms.
ExtractorFn = Callable[["Dataset"], Iterable[str]]


class VocabularyExtractorError(RuntimeError):
    """Raised when a vocabulary extractor function fails during resolution."""


class _Resolved:
    """Cached resolution of a vocabulary against a single dataset + pipeline context."""

    __slots__ = ("terms", "trie")

    def __init__(self, terms: set[KeyTerm], trie: Optional[KeyTermTrie]):
        self.terms = terms
        self.trie = trie


class Vocabulary:
    """A named collection of key terms, defined by any mix of static lists, files, and
    extractor functions, that also performs key term matching.

    A vocabulary holds only its *definition*; the concrete key terms are resolved lazily
    per dataset (extractor functions read the dataset's examples, and static terms are
    tokenized with the dataset's pipelines). Resolution and the resulting trie are cached
    per dataset, so a single Vocabulary can be attached to and shared across multiple
    datasets via :meth:`Dataset.add_vocabulary`.
    """

    def __init__(self, name: str):
        """Initialize an empty vocabulary.

        Args:
            name: The vocabulary name, used to attach it to a dataset, reference it from
                metrics (``metrics.ktr(vocab=name)``), and label it in reports.
        """
        if not isinstance(name, str):
            raise TypeError(f"Vocabulary name must be a string, got {type(name)}.")
        self._name = name
        self._static_terms: set[str] = set()
        self._extractors: list[ExtractorFn] = []
        # Per-dataset resolution cache. Weak keys so an attached-but-discarded dataset's
        # resolution (and trie) is collected with it — a shared vocabulary never leaks.
        self._cache: "weakref.WeakKeyDictionary[Dataset, dict[tuple, _Resolved]]" = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """The vocabulary name."""
        return self._name

    @property
    def has_sources(self) -> bool:
        """Whether the vocabulary has any term source (static terms or extractors)."""
        return bool(self._static_terms or self._extractors)

    # ---- builder ---------------------------------------------------------------------

    def add_terms(self, terms: Iterable[str]) -> "Vocabulary":
        """Add a static list of key terms. Returns self for chaining."""
        if not isinstance(terms, Iterable) or isinstance(terms, str):
            raise TypeError("terms must be an iterable of strings")
        terms = set(terms)
        for term in terms:
            if not isinstance(term, str):
                raise TypeError(f"terms must be an iterable of strings, but got element of type {type(term)}")
        self._static_terms.update(terms)
        return self

    def add_file(self, path: str | Path) -> "Vocabulary":
        """Add key terms from a plain-text file (one term per line). Returns self for chaining."""
        if not Path(path).is_file():
            raise FileNotFoundError(f"Key term file {path} not found")
        terms = Path(path).read_text().strip().splitlines()
        return self.add_terms(terms)

    def add_extractor(self, fn: ExtractorFn) -> "Vocabulary":
        """Add an extractor function ``Callable[[Dataset], Iterable[str]]`` whose output
        key terms are resolved against each dataset the vocabulary is used with.
        Returns self for chaining."""
        if not callable(fn):
            raise TypeError("extractor must be callable")
        self._extractors.append(fn)
        return self

    # ---- resolution / matching -------------------------------------------------------

    def _resolve_terms(self, dataset: "Dataset") -> set[KeyTerm]:
        """Resolve all sources to a deduplicated set of KeyTerm objects for ``dataset``."""
        raw: set[str] = set(self._static_terms)
        for fn in self._extractors:
            try:
                produced = fn(dataset)
            except Exception as e:
                raise VocabularyExtractorError(f"Extractor for vocabulary '{self._name}' raised an error.") from e
            if produced is None:
                continue
            for term in produced:
                if not isinstance(term, str):
                    raise TypeError(
                        f"Extractor for vocabulary '{self._name}' must yield strings, "
                        f"but got element of type {type(term)}."
                    )
                raw.add(term)
        # KeyTerm hashes by (raw, text_type), so the set dedups extracted terms against
        # static ones by raw string — extracted terms become indistinguishable from static.
        return {KeyTerm(term, pipelines=dataset.pipelines) for term in raw}

    def _ctx_key(self, normalized: bool, add_capitalized: bool) -> tuple:
        return (
            STANDARDIZER_NAME.get(),
            TOKENIZER_NAME.get(),
            NORMALIZER_NAME.get() if normalized else None,
            add_capitalized,
        )

    def _get_trie(self, dataset: "Dataset", normalized: bool, add_capitalized: bool) -> Optional[KeyTermTrie]:
        """Get (or lazily build and cache) the trie for ``dataset`` under the active pipeline context."""
        # Freezing here guarantees extractors see the complete dataset and that the
        # resolved term set / trie can never go stale afterwards.
        dataset.freeze()
        key = self._ctx_key(normalized, add_capitalized)
        ds_cache = self._cache.get(dataset)
        if ds_cache is not None and key in ds_cache:
            return ds_cache[key].trie
        with self._lock:
            ds_cache = self._cache.setdefault(dataset, {})
            if key in ds_cache:
                return ds_cache[key].trie
            terms = self._resolve_terms(dataset)
            trie = KeyTermTrie(terms, normalized=normalized, add_capitalized=add_capitalized) if terms else None
            ds_cache[key] = _Resolved(terms, trie)
            return trie

    def find_matches(
        self,
        text: "Text",
        *,
        normalized: bool = True,
        add_capitalized: bool = False,
        allow_subset_matches: bool = False,
    ) -> list[slice]:
        """Find key term matches for this vocabulary in ``text``'s tokens.

        Returns a list of slices representing matched token spans. Returns ``[]`` if the
        text has no parent dataset or the vocabulary resolves to no terms.
        """
        example = text.src
        dataset = example.src if example is not None else None
        if dataset is None:
            return []
        trie = self._get_trie(dataset, normalized, add_capitalized)
        if trie is None:
            return []
        matches, _ = trie.find_in_tokens(text.tokens)
        if not matches:
            return []
        return _remove_duplicate_matches(matches) if allow_subset_matches else _remove_subset_matches(matches)

    def __repr__(self):
        return f"Vocabulary({self._name!r})"
