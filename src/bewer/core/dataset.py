from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Union

import pandas as pd

from bewer.config import BewerConfig, get_config, resolve_config
from bewer.core.example import Example
from bewer.core.text import TokenList
from bewer.core.vocabulary import Vocabulary
from bewer.metrics.base import MetricCollection
from bewer.registry import REGISTRY

__all__ = ["Dataset", "DatasetFrozenError", "TextList", "TextTokenList"]

if TYPE_CHECKING:
    from bewer.core.text import Text


class DatasetFrozenError(RuntimeError):
    """Raised when attempting to modify a Dataset after it has been frozen."""


class Dataset(object):
    """BeWER dataset.

    Attributes:
        config (BewerConfig): The resolved configuration object.
        pipelines: The resolved preprocessing pipelines.
        examples (list[Example]): A list of Example objects.
        metrics (MetricCollection): A metrics collection for the dataset.
        refs (TextList): The reference texts in the dataset.
        hyps (TextList): The hypothesis texts in the dataset.
    """

    def __init__(
        self,
        config: BewerConfig | str | None = None,
        *,
        vocabularies: list[str] | None = None,
    ):
        """Initialize the Dataset.

        The dataset must be populated using one of the load_* methods or manually using the add() method.

        Args:
            config (BewerConfig | str | None): A BewerConfig, a registered config
                name (e.g. "base", "en", "da"), or None for the base config.
            vocabularies (list[str] | None): Names of registered vocabularies to
                attach eagerly at init time.
        """
        if isinstance(config, BewerConfig):
            self._config = config
        elif isinstance(config, str):
            self._config = get_config(config)
        elif config is None:
            self._config = get_config("base")
        else:
            raise TypeError(f"config must be BewerConfig, str, or None, got {type(config)}")

        self.config_path = None

        self._init_blank_state()
        self._pipelines = resolve_config(self._config)

        # Eagerly resolve declared vocabularies
        for name in self._config.vocabularies:
            self._resolve_vocabulary(name)
        for name in vocabularies or []:
            self._resolve_vocabulary(name)

    def _init_blank_state(self) -> None:
        """Initialize the mutable, modifiable state of the dataset.

        Shared by __init__ and clone(). Assumes config and _pipelines are already
        set (or will be set immediately after). Leaves the dataset unfrozen with
        empty data and clean caches.
        """
        self.examples = []
        self._vocabularies: dict[str, "Vocabulary"] = {}
        self.metrics = MetricCollection(self)
        self._frozen = False

    @property
    def pipelines(self):
        return self._pipelines

    @property
    def is_frozen(self) -> bool:
        """Whether the dataset is frozen (immutable). Frozen datasets reject new data."""
        return self._frozen

    def freeze(self) -> None:
        """Freeze the dataset, preventing any further modification.

        Called automatically the first time a metric is requested. Idempotent. To keep
        modifying the data after freezing, use clone() to get a fresh, modifiable copy.
        """
        self._frozen = True

    def _check_not_frozen(self) -> None:
        """Raise if the dataset is frozen. Guards all data-mutating methods."""
        if self._frozen:
            raise DatasetFrozenError(
                "Cannot modify a frozen Dataset: metric computation has begun. "
                "Use Dataset.clone() to get a fresh, modifiable copy."
            )

    def clone(self) -> "Dataset":
        """Return a fresh, unfrozen copy of the dataset with clean caches.

        The copy shares the resolved configuration and preprocessing pipelines (both
        read-only at runtime) but rebuilds all examples, vocabularies and caches from
        scratch, so it carries none of the original's cached metric results. Works
        regardless of whether the original is frozen.

        Returns:
            Dataset: A modifiable copy containing the same examples and key term vocabularies.
        """
        new = object.__new__(Dataset)
        new._config = self._config
        new.config_path = self.config_path
        new._pipelines = self._pipelines
        new._init_blank_state()
        for example in self.examples:
            new.add(example.ref.raw, example.hyp.raw)
        for vocab in self._vocabularies.values():
            new.add_vocabulary(vocab)
        return new

    @cached_property
    def refs(self) -> "TextList":
        """Get the reference texts as a TextList object.

        Returns:
            TextList: The reference texts.
        """
        return TextList([example.ref for example in self.examples])

    @cached_property
    def hyps(self) -> "TextList":
        """Get the hypothesis texts as a TextList object.

        Returns:
            TextList: The hypothesis texts.
        """
        return TextList([example.hyp for example in self.examples])

    def add(self, ref: str, hyp: str) -> None:
        """Add an example to the dataset."""
        self._check_not_frozen()
        example = Example(ref, hyp, pipelines=self._pipelines, src=self, index=len(self))
        self.examples.append(example)
        # Invalidate cached refs/hyps so they stay fresh while the dataset is still being built.
        self.__dict__.pop("refs", None)
        self.__dict__.pop("hyps", None)

    def load_dataset(self, dataset, ref_col="ref", hyp_col="hyp") -> None:
        """Load a Hugging Face dataset."""
        self._check_not_frozen()
        raise NotImplementedError("load_dataset() method not implemented.")

    def load_pandas(self, df: pd.DataFrame, ref_col="ref", hyp_col="hyp") -> None:
        """Add a pandas DataFrame to the dataset."""
        self._check_not_frozen()
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        # Add examples to the dataset
        for row in df.itertuples(index=False):
            hyp = getattr(row, hyp_col)
            ref = getattr(row, ref_col)
            self.add(ref, hyp)

    def load_csv(self, csv_file: str, ref_col="ref", hyp_col="hyp", **kwargs) -> None:
        """Add a CSV file to the dataset."""
        self._check_not_frozen()
        df = pd.read_csv(csv_file, **kwargs)
        self.load_pandas(df, ref_col, hyp_col)

    def load_jsonl(self, jsonl_file: str, ref_col="ref", hyp_col="hyp", **kwargs) -> None:
        """Add a JSONL file to the dataset."""
        self._check_not_frozen()
        df = pd.read_json(jsonl_file, lines=True, **kwargs)
        self.load_pandas(df, ref_col, hyp_col)

    def add_vocabulary(self, vocab: "Vocabulary") -> None:
        """Attach a key term vocabulary to the dataset.

        The vocabulary is registered under its own ``name`` and can be referenced by key
        term metrics via ``metrics.ktr(vocab=name)``. The same Vocabulary object may be
        attached to multiple datasets; it resolves its terms against each one independently.

        Attaching freezes the vocabulary's definition: it can no longer be modified via
        ``add_terms``/``add_file``/``add_extractor``, so its resolved terms stay fixed for
        every dataset it is attached to.

        Args:
            vocab (Vocabulary): The vocabulary to attach.

        Raises:
            TypeError: If ``vocab`` is not a Vocabulary.
            ValueError: If a different vocabulary is already registered under the same name.
        """
        self._check_not_frozen()
        if not isinstance(vocab, Vocabulary):
            raise TypeError(f"add_vocabulary() expects a Vocabulary, got {type(vocab)}.")
        existing = self._vocabularies.get(vocab.name)
        if existing is not None and existing is not vocab:
            raise ValueError(f"A different vocabulary named '{vocab.name}' is already attached to this dataset.")
        self._vocabularies[vocab.name] = vocab
        vocab._freeze()

    def _register_derived_vocabulary(self, vocab: "Vocabulary") -> "Vocabulary":
        """Register a metric-derived vocabulary, returning the one now bound to its name.

        Metric-derived vocabularies (e.g. the auto-extracted ``orthographically_complex_terms``
        backing the orthographically-complex-term metrics) are attached lazily the first time such a
        metric is requested,
        which may happen after the dataset has frozen on an earlier metric. Because the
        vocabulary introduces a brand-new name, it cannot change a term set any prior metric
        already resolved, so registering it on a frozen dataset cannot stale a cached result.
        This therefore bypasses the frozen guard that :meth:`add_vocabulary` enforces.

        If a vocabulary is already registered under the name it is returned unchanged.

        Args:
            vocab (Vocabulary): The derived vocabulary to register.

        Returns:
            Vocabulary: The vocabulary now registered under ``vocab.name``.

        Raises:
            TypeError: If ``vocab`` is not a Vocabulary.
        """
        if not isinstance(vocab, Vocabulary):
            raise TypeError(f"_register_derived_vocabulary() expects a Vocabulary, got {type(vocab)}.")
        existing = self._vocabularies.get(vocab.name)
        if existing is not None:
            return existing
        self._vocabularies[vocab.name] = vocab
        vocab._freeze()
        return vocab

    def _resolve_vocabulary(self, name: str) -> "Vocabulary":
        """Resolve a vocabulary by name — attached first, then registered.

        If the vocabulary is already attached, return it.
        If it is registered in ``REGISTRY.vocabularies``, attach and return it.
        Otherwise raise ValueError with the available names.
        """
        if name in self._vocabularies:
            return self._vocabularies[name]
        if name in REGISTRY.vocabularies:
            vocab = REGISTRY.vocabularies.get(name)
            self._register_derived_vocabulary(vocab)
            return vocab
        raise ValueError(
            f"Vocabulary '{name}' not found. "
            f"Attached: {sorted(self._vocabularies)}, "
            f"Registered: {REGISTRY.vocabularies.list()}"
        )

    @property
    def config(self) -> BewerConfig:
        return self._config

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        """Get an example by index."""
        return self.examples[index]

    def __iter__(self):
        """Iterate over the examples in the dataset."""
        return iter(self.examples)

    def __repr__(self):
        """Get a string representation of the dataset."""
        return f"Dataset({len(self.examples)} examples)"


class TextList(tuple["Text", ...]):
    """An immutable sequence of Text objects."""

    def __new__(cls, iterable=()):
        return super().__new__(cls, iterable)

    @property
    def raw(self) -> list[str]:
        """Get the raw texts as a regular Python list.

        Returns:
            list[str]: The raw text.
        """
        return [text.raw for text in self]

    @property
    def standardized(self) -> list[str]:
        """Get the standardized texts as a regular Python list.

        Returns:
            list[str]: The standardized texts.
        """
        return [text.standardized for text in self]

    @property
    def tokens(self) -> "TextTokenList":
        """Get the tokens as a TextTokenList object.

        Returns:
            TextTokenList: The tokens.
        """
        return TextTokenList([text.tokens for text in self])

    def __getitem__(self, index: int | slice) -> Union["Text", "TextList"]:
        if isinstance(index, slice):
            return TextList(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: "TextList") -> "TextList":
        return TextList(super().__add__(other))

    def __repr__(self):
        texts = self[:60]
        texts_str = ",\n ".join([repr(text) for text in texts])
        if len(self) > 60:
            texts_str += ",\n ..."
        return f"TextList([\n {texts_str}]\n)"


class TextTokenList(tuple["TokenList", ...]):
    """An immutable sequence of TokenList objects."""

    def __new__(cls, iterable=()):
        return super().__new__(cls, iterable)

    @property
    def standardized(self) -> list[list[str]]:
        """Get the standardized tokens as a regular Python list.

        Returns:
            list[list[str]]: The standardized tokens.
        """
        return [tokens.standardized for tokens in self]

    @property
    def normalized(self) -> list[list[str]]:
        """Get the normalized tokens as a regular Python list.

        Returns:
            list[list[str]]: The normalized tokens.
        """
        return [tokens.normalized for tokens in self]

    @property
    def flat(self) -> "TokenList":
        """Flatten the TextTokenList into a TokenList.

        Returns:
            TokenList: The flattened TokenList.
        """
        return TokenList(chain(*self))

    def __getitem__(self, index: int | slice) -> Union["TokenList", "TextTokenList"]:
        if isinstance(index, slice):
            return TextTokenList(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: "TextTokenList") -> "TextTokenList":
        return TextTokenList(super().__add__(other))

    def __repr__(self):
        text_tokens = self[:60]
        text_tokens_str = ",\n ".join([tokens._sub_repr() for tokens in text_tokens])
        if len(self) > 60:
            text_tokens_str += ",\n ..."
        return f"TextTokenList([\n {text_tokens_str}]\n)"
