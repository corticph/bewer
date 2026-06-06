from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

from bewer.core.text import Text, TextType
from bewer.metrics.base import ExampleMetricCollection

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset
    from bewer.core.key_term import KeyTerm

__all__ = ["Example"]


class Example:
    """
    BeWER example representation.

    Attributes:
        ref (Text): Reference text object.
        hyp (Text): Hypothesis text object.
        key_terms (dict[str, set[KeyTerm]]): Resolved key terms grouped by vocabulary name,
            filtered to the canonical terms this example regards.
        metrics (ExampleMetricCollection): Metrics collection for this example.
    """

    def __init__(
        self,
        ref: str,
        hyp: str,
        key_terms: dict[str, Iterable[str]] | None = None,
        *,
        src: Dataset,
        index: Optional[int] = None,
    ):
        """
        Initialize the Example object.

        Args:
            ref: Reference text.
            hyp: Hypothesis text.
            key_terms: Key terms associated with the example, grouped by vocabulary name. The raw
                strings are retained; the canonical KeyTerm objects are resolved (and deduped) by
                the owning Vocabulary. A warning is logged during matching if a regarded term cannot
                be found in the reference tokens.
            src: Parent Dataset object (required).
            index: The index of the example in the dataset.
        """
        self._index = index

        self._src = src
        self._pipelines = src.pipelines

        # Raw per-vocabulary annotation strings. The single source of truth for resolved
        # (canonical) terms is the Vocabulary; `key_terms` below is a derived view.
        self._key_term_strings = self._prepare_key_term_strings(key_terms)

        self.metrics = ExampleMetricCollection(self)
        self.ref = Text(ref, src=self, text_type=TextType.REF)
        self.hyp = Text(hyp, src=self, text_type=TextType.HYP)

    @property
    def index(self) -> Optional[int]:
        """Get the example index."""
        return self._index

    @property
    def src(self) -> Dataset:
        """Get the parent Dataset object."""
        return self._src

    @property
    def pipelines(self):
        return self._pipelines

    @property
    def key_terms(self) -> dict[str, set[KeyTerm]]:
        """The canonical key terms this example regards, grouped by vocabulary name.

        Derived from each named vocabulary's resolved terms, filtered to those whose
        ``examples`` back-reference includes this example. Vocabularies the example
        annotates but contributes no resolved terms to are omitted.
        """
        result: dict[str, set[KeyTerm]] = {}
        for name in self._key_term_strings:
            if not self._src.has_vocabulary(name):
                continue
            vocab = self._src.get_vocabulary(name)
            terms = {kt for kt in vocab.key_terms if self in kt.examples}
            if terms:
                result[name] = terms
        return result

    @property
    def vocabs(self) -> set[str]:
        """Get the set of all key term vocabularies associated with this example."""
        vocabs = set(self._key_term_strings.keys())
        vocabs.update(self._src._vocabularies.keys())
        return vocabs

    @staticmethod
    def _prepare_key_term_strings(key_terms: dict[str, Iterable[str]] | None) -> dict[str, set[str]]:
        """Store the raw per-vocabulary annotation strings (empty groups retained)."""
        if key_terms is None:
            return {}
        return {vocab_name: set(terms) for vocab_name, terms in key_terms.items()}

    def __hash__(self):
        return hash((self.ref, self.hyp, self._index))

    def __repr__(self):
        ref = self.ref.raw if len(self.ref.raw) <= 45 else self.ref.raw[:42] + "..."
        hyp = self.hyp.raw if len(self.hyp.raw) <= 45 else self.hyp.raw[:42] + "..."
        return f'Example(ref="{ref}", hyp="{hyp}")'
