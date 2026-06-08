from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Mapping, Optional, cast

from bewer.core.key_term import (
    KeyTerm,
    KeyTermTrie,
    Match,
    _remove_duplicate_matches,
    _remove_subset_matches,
)
from bewer.core.text import TextType
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.example import Example
    from bewer.core.text import Text

__all__ = ["Vocabulary", "VocabularyExtractor"]

logger = logging.getLogger(__name__)


# A vocabulary extractor derives key terms from a dataset on demand. It returns a mapping of
# example index -> terms; each term is associated with the given example. An empty mapping is
# a no-op.
VocabularyExtractor = Callable[["Dataset"], "Mapping[int, Iterable[str]]"]


class Vocabulary:
    """A named lexicon of key terms that can locate its terms within a text.

    A vocabulary holds *sources* of terms — explicit enumeration (:meth:`from_list`,
    :meth:`from_file`) and lazy extraction by a function (:meth:`from_function`) — plus the
    per-example annotations (the ``key_terms`` passed when examples are added) that name it.
    The terms themselves are canonical :class:`KeyTerm` objects owned by the dataset (see
    :attr:`Dataset.key_terms`); :attr:`key_terms` is this lexicon's slice of them. A vocabulary
    does not hold the example-regard links — those live on the canonical term.

    Whether a term is matched everywhere or only within the examples that regard it is a
    *matching-time policy* controlled by the ``only_local_matches`` flag of :meth:`find_in`.
    A single Aho-Corasick trie is built over the lexicon; local scoping is a post-hoc filter.

    Vocabularies are constructed detached from any dataset; the back-reference is wired when
    the vocabulary is registered via :meth:`Dataset.add_vocabulary`. Vocabularies are
    *combinable*: :meth:`combine` (and ``+``) unions their sources into a new detached
    vocabulary (names need not match); registering a vocabulary whose name already exists
    folds its sources into the existing instance.
    """

    def __init__(
        self,
        name: str,
        terms: Optional[Iterable[str]] = None,
        extractor: Optional[VocabularyExtractor] = None,
    ):
        """Initialize the vocabulary.

        Args:
            name: The vocabulary name.
            terms: Explicit (global) key term strings. May be combined with ``extractor`` and
                with per-example annotations.
            extractor: A function ``(dataset) -> mapping`` run lazily to extract additional
                terms. May be combined with ``terms``; the final key term set is their union.
        """
        self.name = name
        self._dataset: Optional[Dataset] = None
        self._explicit_terms: set[str] = set(terms) if terms is not None else set()
        # Lazily-run extractors; a vocabulary may accumulate several when same-name
        # vocabularies are combined (see :meth:`combine`). All are run and their terms unioned.
        self._extractors: list[VocabularyExtractor] = [extractor] if extractor is not None else []
        # Names under which per-example annotations feed this lexicon. Just its own name until
        # combined: a combined vocabulary draws annotations from all its constituents' names.
        self._annotation_names: set[str] = {name}
        # Matching caches keyed by pipeline state. The trie is keyed by pipeline only; matches
        # are additionally keyed by example, text type, and the matching flags.
        self._trie_cache: dict[tuple, Optional[KeyTermTrie]] = {}
        self._match_cache: dict[tuple, list[Match]] = {}

    @property
    def dataset(self) -> Optional[Dataset]:
        """The dataset this vocabulary is registered with, if any."""
        return self._dataset

    @classmethod
    def from_list(cls, name: str, terms: Iterable[str]) -> Vocabulary:
        """Create a vocabulary from a list of key term strings."""
        if isinstance(terms, str) or not isinstance(terms, Iterable):
            raise TypeError("terms must be an iterable of strings")
        terms = set(terms)
        for term in terms:
            if not isinstance(term, str):
                raise TypeError(f"terms must be an iterable of strings, but got element of type {type(term)}")
        return cls(name, terms=terms)

    @classmethod
    def from_file(cls, name: str, path: str | Path) -> Vocabulary:
        """Create a vocabulary from a file with one key term per line."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Key term file {path} not found")
        with path.open("r") as f:
            terms = [line.strip() for line in f if line.strip()]
        return cls.from_list(name, terms)

    @classmethod
    def from_function(cls, name: str, fn: VocabularyExtractor) -> Vocabulary:
        """Create a vocabulary whose terms are extracted lazily by ``fn(dataset)``."""
        if not callable(fn):
            raise TypeError("fn must be callable")
        return cls(name, extractor=fn)

    def combine(self, other: Vocabulary, *, name: Optional[str] = None) -> Vocabulary:
        """Combine with another vocabulary into a new detached vocabulary named ``name``.

        The result unions both vocabularies' explicit terms, extractors, and the annotation
        names they draw from (so per-example annotations under either constituent's name flow
        into the combined lexicon); ``name`` defaults to this vocabulary's name. The names need
        not match — terms are canonical at the
        dataset level, so a combined vocabulary genuinely contains the same shared :class:`KeyTerm`
        objects (each then listing the combined vocabulary among its ``vocabularies``). The
        result is detached regardless of the operands' binding; it is bound when registered.
        """
        combined = Vocabulary(name if name is not None else self.name)
        combined._merge_sources(self)
        combined._merge_sources(other)
        return combined

    def __add__(self, other: Vocabulary) -> Vocabulary:
        return self.combine(other)

    def _merge_sources(self, other: Vocabulary) -> None:
        """Fold another vocabulary's sources (explicit terms, extractors, annotation names) in place."""
        self._explicit_terms |= other._explicit_terms
        self._extractors += other._extractors
        self._annotation_names |= other._annotation_names

    @property
    def key_terms(self) -> set[KeyTerm]:
        """This lexicon's slice of the dataset's canonical key terms."""
        if self._dataset is None:
            raise ValueError(f"Vocabulary '{self.name}' is not bound to a dataset.")
        return {kt for kt in self._dataset.key_terms.values() if self in kt.vocabularies}

    def _iter_sources(self) -> Iterator[tuple[str, Optional[Example]]]:
        """Yield (raw, example) from this vocabulary's own sources; example is None for explicit terms."""
        for raw in self._explicit_terms:
            yield raw, None
        for position, extractor in enumerate(self._extractors, start=1):
            yield from self._extracted_associations(extractor, position)

    def _extracted_associations(self, extractor: VocabularyExtractor, position: int) -> Iterator[tuple[str, Example]]:
        """Yield (raw, example) pairs from a single extractor's per-example mapping.

        ``position`` is the extractor's 1-based position within :attr:`_extractors`, used only
        to attribute errors to a specific extractor when a vocabulary combines several.
        """
        assert self._dataset is not None
        try:
            extracted = extractor(self._dataset)
        except Exception as e:
            raise RuntimeError(f"Error running extractor {position} for vocabulary '{self.name}': {e}") from e
        if not isinstance(extracted, Mapping):
            raise TypeError(
                f"Extractor {position} for vocabulary '{self.name}' must return a mapping of example index to terms."
            )
        for index, terms in cast(Mapping[int, Iterable[str]], extracted).items():
            example = self._dataset[index]
            for term in terms:
                yield term, example

    def find_in(
        self,
        text: Text,
        *,
        normalized: bool = True,
        add_capitalized: bool = False,
        allow_subset_matches: bool = False,
        only_local_matches: bool = False,
    ) -> list[Match]:
        """Find key term matches in ``text``'s tokens.

        A single trie is built over the union of the vocabulary's terms. When
        ``only_local_matches`` is True, a match is kept only if at least one of its key
        terms is regarded by ``text``'s example (so global-only terms and examples with no
        regarded terms yield nothing); when False, every union term matches everywhere.

        On the reference side, a warning is logged for each term the example regards that is
        absent from the matched terms.

        Args:
            text: The text to match against.
            normalized: Use normalized tokens for matching.
            add_capitalized: Add capitalized first-token variants (raw mode only).
            allow_subset_matches: If False, discard matches whose span is a subset of a longer match.
            only_local_matches: If True, scope matches to terms regarded by this text's example.

        Returns:
            A list of :class:`Match` objects.
        """
        example = text.src
        example_index = example.index if example is not None else None
        pipeline_key = self._pipeline_key(normalized, add_capitalized)
        cache_key = (example_index, text.text_type, allow_subset_matches, only_local_matches) + pipeline_key
        if cache_key in self._match_cache:
            return self._match_cache[cache_key]

        trie = self._get_trie(pipeline_key, normalized, add_capitalized)
        if trie is None:
            self._match_cache[cache_key] = []
            return []

        spans, patterns = trie.find_in_tokens(text.tokens)
        matches = [
            Match(span=span, text=text, key_terms=frozenset(trie.terms_for_pattern(pattern)))
            for span, pattern in zip(spans, patterns)
        ]

        # On the reference side, verify each term the example regards is actually present.
        if text.text_type == TextType.REF and example is not None:
            matched_terms = {kt for match in matches for kt in match.key_terms}
            for key_term in self.key_terms:
                if example in key_term.examples and key_term not in matched_terms:
                    logger.warning(
                        "Key term '%s' not found in reference tokens: Example %s.", key_term.raw, example_index
                    )

        if only_local_matches:
            matches = [m for m in matches if any(example in kt.examples for kt in m.key_terms)]

        matches = self._filter_matches(matches, allow_subset_matches)
        self._match_cache[cache_key] = matches
        return matches

    def _bind(self, dataset: Dataset) -> None:
        """Wire the dataset back-reference. Called by Dataset.add_vocabulary / _ensure_vocabulary."""
        if self._dataset is not None and self._dataset is not dataset:
            raise ValueError(f"Vocabulary '{self.name}' is already bound to a different dataset.")
        self._dataset = dataset

    def invalidate_caches(self) -> None:
        """Drop cached tries and matches so they are rebuilt on next access."""
        self._trie_cache.clear()
        self._match_cache.clear()

    @staticmethod
    def _pipeline_key(normalized: bool, add_capitalized: bool) -> tuple:
        """Build a cache key component from the active preprocessing pipeline."""
        return (
            STANDARDIZER_NAME.get(),
            TOKENIZER_NAME.get(),
            NORMALIZER_NAME.get() if normalized else None,
            add_capitalized,
        )

    @staticmethod
    def _filter_matches(matches: list[Match], allow_subset_matches: bool) -> list[Match]:
        """Deduplicate or remove subset matches according to the flag."""
        if not matches:
            return matches
        if allow_subset_matches:
            return _remove_duplicate_matches(matches)
        return _remove_subset_matches(matches)

    def _get_trie(
        self,
        cache_key: tuple,
        normalized: bool,
        add_capitalized: bool,
    ) -> Optional[KeyTermTrie]:
        """Get or build (and cache) the trie over the union of terms for the active pipeline.

        Cannot be a property: the trie is keyed by the matching options
        (``normalized`` / ``add_capitalized``).
        """
        if cache_key in self._trie_cache:
            return self._trie_cache[cache_key]
        terms = self.key_terms
        trie = KeyTermTrie(terms, normalized=normalized, add_capitalized=add_capitalized) if terms else None
        self._trie_cache[cache_key] = trie
        return trie
