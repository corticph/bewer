from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from bewer.core.key_term import (
    KeyTerm,
    KeyTermTrie,
)

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset

__all__ = ["Vocabulary", "VocabularyExtractorError", "VocabularyFrozenError", "ExtractorFn"]

# An extractor takes a Dataset and returns key term strings pulled from its examples
# (reference or hypothesis text). Extracted terms are treated exactly like static terms.
ExtractorFn = Callable[["Dataset"], Iterable[str]]


class VocabularyExtractorError(RuntimeError):
    """Raised when a vocabulary extractor function fails during resolution."""


class VocabularyFrozenError(RuntimeError):
    """Raised when attempting to modify a Vocabulary after it has been used for matching."""


class Vocabulary:
    """A named collection of key terms, defined by any mix of static lists, files, and
    extractor functions.

    A vocabulary holds only its *definition*; the concrete key terms are resolved lazily
    per dataset (extractor functions read the dataset's examples, and static terms are
    tokenized with the dataset's pipelines). The resolved :class:`KeyTermTrie` is returned
    by :meth:`build_matcher` and cached by the caller (e.g. a metric), not by the
    Vocabulary itself.
    """

    def __init__(self, name: str):
        """Initialize an empty vocabulary.

        Args:
            name: The vocabulary name, used to reference it from metrics
                (``metrics.ktr(vocab=name)``) and label it in reports.
        """
        if not isinstance(name, str):
            raise TypeError(f"Vocabulary name must be a string, got {type(name)}.")
        self._name = name
        self._static_terms: set[str] = set()
        self._extractors: list[ExtractorFn] = []
        # Frozen once used for matching (see build_matcher). A frozen vocabulary's
        # definition cannot change, so its resolved terms / trie can never go stale
        # for any dataset it is used with.
        self._frozen = False

    def _freeze(self) -> None:
        """Mark the vocabulary's definition as immutable. Idempotent. Called lazily
        on first resolution (``build_matcher``), so the definition is locked as soon as
        it is first used against any dataset."""
        self._frozen = True

    def _check_not_frozen(self) -> None:
        if self._frozen:
            raise VocabularyFrozenError(
                f"Cannot modify vocabulary '{self._name}' after it has been used for matching. "
                "Build it fully before using it in a metric, or create a new Vocabulary."
            )

    @property
    def name(self) -> str:
        """The vocabulary name."""
        return self._name

    # ---- builder ---------------------------------------------------------------------

    def add_terms(self, terms: Iterable[str]) -> "Vocabulary":
        """Add a static list of key terms. Returns self for chaining."""
        self._check_not_frozen()
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
        self._check_not_frozen()
        if not Path(path).is_file():
            raise FileNotFoundError(f"Key term file {path} not found")
        terms = Path(path).read_text().strip().splitlines()
        return self.add_terms(terms)

    def add_extractor(self, fn: ExtractorFn) -> "Vocabulary":
        """Add an extractor function ``Callable[[Dataset], Iterable[str]]`` whose output
        key terms are resolved against each dataset the vocabulary is used with.
        Returns self for chaining."""
        self._check_not_frozen()
        if not callable(fn):
            raise TypeError("extractor must be callable")
        self._extractors.append(fn)
        return self

    # ---- resolution ------------------------------------------------------------------

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

    def build_matcher(
        self,
        dataset: "Dataset",
        *,
        normalized: bool = True,
        add_capitalized: bool = False,
    ) -> KeyTermTrie:
        """Resolve the vocabulary against ``dataset`` and return a :class:`KeyTermTrie`.

        Freezes the vocabulary (so its definition cannot change afterwards) and the
        dataset (so extractors see a stable snapshot). A vocabulary with no terms
        returns a trie that matches nothing — ``find_matches`` on it yields ``[]``.

        The caller owns the returned trie — the Vocabulary does not cache it. Metrics
        should cache the trie via ``@metric_property``.
        """
        self._freeze()
        dataset.freeze()
        terms = self._resolve_terms(dataset)
        return KeyTermTrie(terms, normalized=normalized, add_capitalized=add_capitalized)

    def __repr__(self):
        return f"Vocabulary({self._name!r})"
