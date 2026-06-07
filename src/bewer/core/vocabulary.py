from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Mapping, Optional, Union, cast

from bewer.core.caching import pipeline_cached_property
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


# A vocabulary extractor derives key terms from a dataset on demand. Returning an iterable
# of strings yields *global* terms (matched in every example, no example association).
# Returning a mapping of example index -> terms yields *local* terms, each associated with
# the given example. An empty result is treated as a global no-op.
VocabularyExtractor = Callable[["Dataset"], Union[Iterable[str], "Mapping[int, Iterable[str]]"]]


class Vocabulary:
    """A named source of key terms that can locate its terms within a text.

    A vocabulary unifies two concerns: *extraction* (where the key terms come from) and
    *matching* (finding those terms in a text's tokens). Terms may come from three sources,
    combined freely as a union:

    - explicit enumeration (:meth:`from_list`, :meth:`from_file`),
    - lazy extraction by a function (:meth:`from_function`), and
    - per-example annotations (the ``key_terms`` passed when examples are added).

    Whether a term is matched everywhere or only within the examples that regard it is a
    *matching-time policy* controlled by the ``only_local_matches`` flag of :meth:`find_in`,
    not a storage distinction. A single Aho-Corasick trie is built over the union of terms;
    local scoping is a post-hoc filter.

    Vocabularies are constructed detached from any dataset; the back-reference is wired when
    the vocabulary is registered via :meth:`Dataset.add_vocabulary`.
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
            extractor: A function ``(dataset) -> terms`` run lazily to extract additional
                terms. May be combined with ``terms``; the final key term set is their union.
        """
        self.name = name
        self._dataset: Optional[Dataset] = None
        self._explicit_terms: set[str] = set(terms) if terms is not None else set()
        self._extractor: Optional[VocabularyExtractor] = extractor
        # Canonical KeyTerm instances, keyed by raw string (rebuilt on each resolution).
        self._terms_by_raw: dict[str, KeyTerm] = {}
        # Caches keyed by pipeline state. The trie is keyed by pipeline only; matches are
        # additionally keyed by example, text type, and the matching flags.
        self._trie_cache: dict[tuple, Optional[KeyTermTrie]] = {}
        self._match_cache: dict[tuple, list[Match]] = {}
        # Per-pipeline cache for the `key_terms` descriptor (see pipeline_cached_property).
        self._cache_key_terms: dict = {}

    @property
    def dataset(self) -> Optional[Dataset]:
        """The dataset this vocabulary is registered with, if any."""
        return self._dataset

    @property
    def pipelines(self):
        """The bound dataset's preprocessing pipelines."""
        if self._dataset is None:
            raise ValueError(f"Vocabulary '{self.name}' is not bound to a dataset.")
        return self._dataset.pipelines

    # The pipeline caching descriptor reads `instance._pipelines`; alias it to the property.
    _pipelines = pipelines

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

    @pipeline_cached_property(NORMALIZER_NAME)
    def key_terms(self, _normalizer) -> set[KeyTerm]:
        """The vocabulary's key terms, resolved (and cached) per active pipeline.

        ``_normalizer`` is intentionally unused: only the per-pipeline cache key matters,
        since key terms are normalized through the active pipeline when matched.
        """
        return self._resolve_key_terms()

    def _resolve_key_terms(self) -> set[KeyTerm]:
        """Build the canonical KeyTerms (deduped by raw) and wire their example back-refs."""
        if self._dataset is None:
            raise ValueError(f"Vocabulary '{self.name}' is not bound to a dataset.")

        # Collect, per raw string, the set of examples that regard it (empty => global).
        associations: dict[str, set[Example]] = defaultdict(set)
        for raw in self._explicit_terms:
            associations.setdefault(raw, set())
        if self._extractor is not None:
            for raw, example in self._extracted_associations():
                if example is None:
                    associations.setdefault(raw, set())
                else:
                    associations[raw].add(example)
        for example in self._dataset:
            for raw in example._key_term_strings.get(self.name, ()):
                associations[raw].add(example)

        self._terms_by_raw = {}
        key_terms: set[KeyTerm] = set()
        for raw, examples in associations.items():
            key_term = KeyTerm(raw, src=self)
            key_term.examples.update(examples)
            self._terms_by_raw[raw] = key_term
            key_terms.add(key_term)
        return key_terms

    def _extracted_associations(self) -> Iterator[tuple[str, Optional[Example]]]:
        """Yield (raw, example) pairs from the extractor; example is None for global terms."""
        assert self._extractor is not None and self._dataset is not None
        try:
            extracted = self._extractor(self._dataset)
        except Exception as e:
            raise RuntimeError(f"Error running extractor for vocabulary '{self.name}': {e}") from e
        if isinstance(extracted, Mapping):
            for index, terms in cast(Mapping[int, Iterable[str]], extracted).items():
                example = self._dataset[index]
                for term in terms:
                    yield term, example
            return
        if isinstance(extracted, str) or not isinstance(extracted, Iterable):
            raise TypeError(
                f"Extractor for vocabulary '{self.name}' must return an iterable of strings or a "
                f"mapping of example index to terms."
            )
        for term in extracted:
            if not isinstance(term, str):
                raise TypeError(
                    f"Extractor for vocabulary '{self.name}' must return an iterable of strings or a "
                    f"mapping of example index to terms, but got element of type {type(term)}."
                )
            yield term, None

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
        """Drop cached terms, tries, and matches so they are rebuilt on next access."""
        self._trie_cache.clear()
        self._match_cache.clear()
        self._terms_by_raw.clear()
        self._cache_key_terms.clear()

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
