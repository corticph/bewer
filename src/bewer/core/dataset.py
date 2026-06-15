from functools import cached_property
from importlib import resources
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Union

import pandas as pd
from omegaconf import OmegaConf

from bewer.configs.resolve import resolve_pipelines
from bewer.core.example import Example
from bewer.core.text import TokenList
from bewer.core.vocabulary import Vocabulary
from bewer.metrics.base import MetricCollection

__all__ = ["Dataset", "DatasetFrozenError", "TextList", "TextTokenList"]

if TYPE_CHECKING:
    from bewer.core.text import Text


class DatasetFrozenError(RuntimeError):
    """Raised when attempting to modify a Dataset after it has been frozen."""


class Dataset(object):
    """BeWER dataset.

    Attributes:
        config (OmegaConf): The resolved configuration object.
        pipelines: The resolved preprocessing pipelines.
        examples (list[Example]): A list of Example objects.
        metrics (MetricCollection): A metrics collection for the dataset.
        refs (TextList): The reference texts in the dataset.
        hyps (TextList): The hypothesis texts in the dataset.
    """

    def __init__(self, config: str | None = None, language: str | None = None):
        """Initialize the Dataset.

        The dataset must be populated using one of the load_* methods or manually using the add() method.

        Args:
            config (str | None): Path to the configuration file. If None, uses the default configuration.
            language (str | None): Language code to apply language-specific pipeline settings (e.g. "da",
                "de", "fr"). If None, uses the default configuration. Supported languages are determined
                by the files in bewer/configs/languages/.
        """
        self.config_path = self.get_config_path(config)
        self.config = OmegaConf.load(self.config_path)
        if language is not None:
            lang_cfg = OmegaConf.load(self._get_language_config_path(language))
            self.config = OmegaConf.merge(self.config, lang_cfg)
        self._pipelines = resolve_pipelines(self.config)
        self._init_blank_state()

    def _init_blank_state(self) -> None:
        """Initialize the mutable, modifiable state of the dataset.

        Shared by __init__ and clone(). Assumes config_path, config and _pipelines
        are already set. Leaves the dataset unfrozen with empty data and clean caches.
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
        new.config_path = self.config_path
        new.config = self.config.copy()
        new._pipelines = self._pipelines
        new._init_blank_state()
        for example in self.examples:
            new.add(example.ref.raw, example.hyp.raw)
        # Share the same Vocabulary objects; the clone re-resolves them against itself
        # (so extractor-defined vocabularies reflect the clone's own examples).
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
        backing the orthographically-complex-term metrics) are attached lazily the first time such
        a metric is requested,
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

    @staticmethod
    def _get_language_config_path(language: str):
        """Resolve the overlay config path for the given language code."""
        languages_dir = resources.files("bewer.configs").joinpath("languages")
        path = languages_dir.joinpath(f"{language}.yml")
        if not path.is_file():
            supported = [p.name[:-4] for p in languages_dir.iterdir() if p.name.endswith(".yml")]
            raise ValueError(f"Unknown language '{language}'. Supported languages: {sorted(supported)}.")
        return path

    @staticmethod
    def get_config_path(config_path: str | None) -> str:
        """Get the configuration path."""
        if config_path is None or not Path(config_path).is_file():
            config_path = "base" if config_path is None else config_path
            return resources.files("bewer.configs").joinpath(f"{config_path}.yml")
        return Path(config_path).resolve()

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
    def raw(self) -> list[list[str]]:
        """Get the raw tokens as a regular Python list.

        Returns:
            list[list[str]]: The raw tokens.
        """
        return [tokens.raw for tokens in self]

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
