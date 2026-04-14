from typing import TYPE_CHECKING, Optional

from bewer.core.key_term import (
    KeyTerm,
    KeyTermTrie,
    _remove_duplicate_matches,
    _remove_subset_matches,
    get_key_term_trie,
)
from bewer.core.text import Text, TextType
from bewer.metrics.base import ExampleMetricCollection
from bewer.preprocessing.context import NORMALIZER_NAME, STANDARDIZER_NAME, TOKENIZER_NAME

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset

__all__ = ["Example"]


class Example:
    """
    BeWER example representation.

    Attributes:
        ref (Text): Reference text object.
        hyp (Text): Hypothesis text object.
        key_terms (dict[str, set[KeyTerm]]): Key terms grouped by vocabulary name.
        metrics (ExampleMetricCollection): Metrics collection for this example.
    """

    def __init__(
        self,
        ref: str,
        hyp: str,
        key_terms: dict[str, list[str]] | None = None,
        src: Optional["Dataset"] = None,
        index: Optional[int] = None,
    ):
        """
        Initialize the Example object.

        Args:
            ref: Reference text.
            hyp: Hypothesis text.
            key_terms: Key terms associated with the example. Missing terms are retained; warnings are emitted
                during key term trie matching if a term cannot be matched in the reference tokens.
            src: Parent Dataset object. Can be set later via set_source().
            index: The index of the example in the dataset.
        """
        self._index = index
        self._cache_key_term_matches = {}
        self._cache_key_term_tries = {}
        self._pipelines = None

        self._src = None
        if src is not None:
            self.set_source(src)

        self.metrics = ExampleMetricCollection(self)
        self.ref = Text(ref, src=self, text_type=TextType.REF)
        self.hyp = Text(hyp, src=self, text_type=TextType.HYP)
        self.key_terms = self._prepare_key_terms(key_terms)

    @property
    def index(self) -> Optional[int]:
        """Get the example index."""
        return self._index

    @property
    def src(self) -> Optional["Dataset"]:
        """Get the parent Dataset object."""
        return self._src

    @property
    def pipelines(self):
        return self._pipelines

    @property
    def vocabs(self) -> set[str]:
        """Get the set of all key term vocabularies associated with this example."""
        vocabs = set(self.key_terms.keys())
        if self._src is not None:
            vocabs.update(self._src._dynamic_key_term_vocabs.keys())
        return vocabs

    def set_source(self, src: "Dataset") -> None:
        """Set the parent Dataset object.

        Args:
            src: The parent Dataset object.

        Raises:
            ValueError: If source is already set.
        """
        if self._src is not None:
            raise ValueError("Source already set for Example")
        self._src = src
        self._pipelines = src.pipelines

    def _prepare_key_terms(self, key_terms: dict[str, set[str]] | None) -> dict[str, set[KeyTerm]]:
        """Prepare key terms dictionary by converting key terms to KeyTerm objects."""
        if key_terms is None:
            return {}

        prepared_key_terms = {}
        for vocab_name, vocab_key_terms in key_terms.items():
            if len(vocab_key_terms) == 0:
                continue
            prepared_key_terms[vocab_name] = set(KeyTerm(key_term, src=self) for key_term in vocab_key_terms)

        return prepared_key_terms

    def _get_key_term_trie(
        self,
        vocab: str,
        normalized: bool = True,
        add_capitalized: bool = False,
    ) -> Optional["KeyTermTrie"]:
        """Get or build a trie for the key terms in the specified vocabulary."""
        return get_key_term_trie(
            self.key_terms,
            self._cache_key_term_tries,
            vocab,
            normalized=normalized,
            add_capitalized=add_capitalized,
        )

    def get_key_term_matches(
        self,
        vocab: str,
        normalized: bool = True,
        add_capitalized: bool = False,
        allow_subsets: bool = True,
        side: TextType = TextType.REF,
    ) -> list[slice]:
        if vocab not in self.key_terms and vocab not in self.src._dynamic_key_term_vocabs:
            return []

        cache_key = (
            STANDARDIZER_NAME.get(),
            TOKENIZER_NAME.get(),
            NORMALIZER_NAME.get() if normalized else None,
            add_capitalized,
            allow_subsets,
            vocab,
            side,
        )
        if cache_key in self._cache_key_term_matches:
            return self._cache_key_term_matches[cache_key]

        tokens = self.ref.tokens if side == TextType.REF else self.hyp.tokens

        matches = []
        if vocab in self.key_terms:
            example_trie = self._get_key_term_trie(vocab, normalized=normalized, add_capitalized=add_capitalized)
            matches += example_trie.find_in_tokens(tokens, warn_missing=(side == TextType.REF))

        if vocab in self.src._dynamic_key_term_vocabs:
            dataset_trie = self.src._get_key_term_trie(vocab, normalized=normalized, add_capitalized=add_capitalized)
            if dataset_trie is not None:
                matches += dataset_trie.find_in_tokens(tokens)

        if matches:
            if allow_subsets:
                matches = _remove_duplicate_matches(matches)
            else:
                matches = _remove_subset_matches(matches)

        self._cache_key_term_matches[cache_key] = matches
        return matches

    def __hash__(self):
        return hash((self.ref, self.hyp, self._index))

    def __repr__(self):
        ref = self.ref.raw if len(self.ref.raw) <= 45 else self.ref.raw[:42] + "..."
        hyp = self.hyp.raw if len(self.hyp.raw) <= 45 else self.hyp.raw[:42] + "..."
        return f'Example(ref="{ref}", hyp="{hyp}")'
