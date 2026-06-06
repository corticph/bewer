from __future__ import annotations

import ast
from functools import cached_property
from importlib import resources
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Union, overload

import pandas as pd
from omegaconf import OmegaConf

from bewer.configs.resolve import resolve_pipelines
from bewer.core.example import Example
from bewer.core.text import TokenList
from bewer.core.vocabulary import Vocabulary, VocabularyExtractor
from bewer.metrics.base import MetricCollection

__all__ = ["Dataset", "TextList", "TextTokenList"]

if TYPE_CHECKING:
    from bewer.core.text import Text


def _is_list_literal(s):
    try:
        return isinstance(ast.literal_eval(s), list)
    except (ValueError, SyntaxError):
        return False


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
        self.examples = []
        self._vocabularies: dict[str, Vocabulary] = {}
        self.metrics = MetricCollection(self)

    @property
    def pipelines(self):
        return self._pipelines

    @cached_property
    def refs(self) -> TextList:
        """Get the reference texts as a TextList object.

        Returns:
            TextList: The reference texts.
        """
        return TextList([example.ref for example in self.examples])

    @cached_property
    def hyps(self) -> TextList:
        """Get the hypothesis texts as a TextList object.

        Returns:
            TextList: The hypothesis texts.
        """
        return TextList([example.hyp for example in self.examples])

    def add(self, ref: str, hyp: str, key_terms: dict[str, Iterable[str]] | None = None) -> None:
        """Add an example to the dataset.

        Any vocabulary names in ``key_terms`` are registered (if not already present); the
        annotated terms become part of those vocabularies and are associated with this
        example. By default a term matches in every example's text; pass
        ``local_only_matches=True`` when matching to scope terms to the examples that
        regard them.
        """
        self._add_example(ref, hyp, key_terms=key_terms)
        self._invalidate_caches()

    def _add_example(self, ref: str, hyp: str, key_terms: dict[str, Iterable[str]] | None = None) -> None:
        """Append a single example without invalidating caches (used for bulk loading)."""
        if key_terms is not None:
            key_terms = {name: set(kt_list) for name, kt_list in key_terms.items()}
            for name in key_terms:
                self._ensure_vocabulary(name)
        example = Example(ref, hyp, key_terms=key_terms, src=self, index=len(self))
        self.examples.append(example)

    def load_dataset(self, dataset, ref_col="ref", hyp_col="hyp", key_term_cols: list | None = None) -> None:
        """Load a Hugging Face dataset."""
        raise NotImplementedError("load_dataset() method not implemented.")

    def load_pandas(self, df: pd.DataFrame, ref_col="ref", hyp_col="hyp", key_term_cols: list | None = None) -> None:
        """Add a pandas DataFrame to the dataset."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if key_term_cols is None:
            key_term_cols = []

        for col in key_term_cols:
            df[col] = self._infer_key_term_column(df[col])

        # Add examples to the dataset (invalidate caches once after the bulk load)
        for row in df.itertuples(index=False):
            hyp = getattr(row, hyp_col)
            ref = getattr(row, ref_col)
            if len(key_term_cols) > 0:
                key_terms = {}
                for col in key_term_cols:
                    key_terms[col] = getattr(row, col)
            else:
                key_terms = None
            self._add_example(ref, hyp, key_terms=key_terms)
        self._invalidate_caches()

    def load_csv(
        self, csv_file: str, ref_col="ref", hyp_col="hyp", key_term_cols: list | None = None, **kwargs
    ) -> None:
        """Add a CSV file to the dataset."""
        df = pd.read_csv(csv_file, **kwargs)
        self.load_pandas(df, ref_col, hyp_col, key_term_cols)

    def load_jsonl(
        self, jsonl_file: str, ref_col="ref", hyp_col="hyp", key_term_cols: list | None = None, **kwargs
    ) -> None:
        """Add a JSONL file to the dataset."""
        df = pd.read_json(jsonl_file, lines=True, **kwargs)
        self.load_pandas(df, ref_col, hyp_col, key_term_cols)

    def add_vocabulary(self, vocabulary: Vocabulary) -> Vocabulary:
        """Register a pre-built vocabulary with the dataset.

        Args:
            vocabulary: The vocabulary to register.

        Returns:
            The registered vocabulary.

        Raises:
            ValueError: If a vocabulary with the same name is already registered.
        """
        if vocabulary.name in self._vocabularies:
            raise ValueError(f"Vocabulary '{vocabulary.name}' is already registered.")
        vocabulary._bind(self)
        self._vocabularies[vocabulary.name] = vocabulary
        self._invalidate_caches()
        return vocabulary

    def add_vocabulary_from_list(self, name: str, terms: Iterable[str]) -> Vocabulary:
        """Add a vocabulary from a list of key term strings.

        Terms are matched against every example's text (including examples already added).
        """
        return self.add_vocabulary(Vocabulary.from_list(name, terms))

    def add_vocabulary_from_file(self, name: str, path: str | Path) -> Vocabulary:
        """Add a vocabulary from a file with one key term per line."""
        return self.add_vocabulary(Vocabulary.from_file(name, path))

    def add_vocabulary_from_function(self, name: str, fn: VocabularyExtractor) -> Vocabulary:
        """Add a vocabulary whose terms are extracted lazily by ``fn(dataset)``.

        The extractor is called the first time the vocabulary's terms are needed and its
        result is cached until the dataset is mutated.
        """
        return self.add_vocabulary(Vocabulary.from_function(name, fn))

    def get_vocabulary(self, name: str) -> Vocabulary:
        """Get a registered vocabulary by name.

        Raises:
            ValueError: If no vocabulary with that name is registered.
        """
        if name not in self._vocabularies:
            raise ValueError(f"Vocabulary '{name}' not found in dataset key term vocabularies.")
        return self._vocabularies[name]

    def has_vocabulary(self, name: str) -> bool:
        """Return whether a vocabulary with the given name is registered."""
        return name in self._vocabularies

    def _ensure_vocabulary(self, name: str) -> Vocabulary:
        """Get or create the vocabulary for a per-example key term annotation name.

        A vocabulary may hold explicit terms and per-example annotations together, so an
        existing vocabulary (however it was created) is simply reused.
        """
        existing = self._vocabularies.get(name)
        if existing is not None:
            return existing
        vocabulary = Vocabulary(name)
        vocabulary._bind(self)
        self._vocabularies[name] = vocabulary
        return vocabulary

    def _invalidate_caches(self) -> None:
        """Invalidate cached terms, tries, matches, and metrics after a mutation."""
        self.__dict__.pop("refs", None)
        self.__dict__.pop("hyps", None)
        for vocabulary in self._vocabularies.values():
            vocabulary.invalidate_caches()
        self.metrics._metric_cache.clear()
        for example in self.examples:
            example.metrics._cache.clear()

    @staticmethod
    def _get_language_config_path(language: str) -> Path:
        """Resolve the overlay config path for the given language code."""
        languages_dir = resources.files("bewer.configs").joinpath("languages")
        path = languages_dir.joinpath(f"{language}.yml")
        if not path.is_file():
            supported = [p.name[:-4] for p in languages_dir.iterdir() if p.name.endswith(".yml")]
            raise ValueError(f"Unknown language '{language}'. Supported languages: {sorted(supported)}.")
        return Path(str(path))

    @staticmethod
    def get_config_path(config_path: str | None) -> Path:
        """Get the configuration path."""
        if config_path is None or not Path(config_path).is_file():
            config_path = "base" if config_path is None else config_path
            return Path(str(resources.files("bewer.configs").joinpath(f"{config_path}.yml")))
        return Path(config_path).resolve()

    def _infer_key_term_column(self, series: pd.Series) -> pd.Series:
        """Infer the key terms from a pandas Series."""
        if series.map(_is_list_literal).all():
            series = series.apply(ast.literal_eval)
            return series
        elif series.map(lambda x: isinstance(x, str)).all():
            series = series.apply(lambda x: [x])
            return series
        elif series.map(lambda x: isinstance(x, list)).all():
            return series
        else:
            raise ValueError(f"Column {series.name} is not a list (or literal) or string")

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
    def tokens(self) -> TextTokenList:
        """Get the tokens as a TextTokenList object.

        Returns:
            TextTokenList: The tokens.
        """
        return TextTokenList([text.tokens for text in self])

    @overload
    def __getitem__(self, index: int) -> Text: ...

    @overload
    def __getitem__(self, index: slice) -> TextList: ...

    def __getitem__(self, index: int | slice) -> Union[Text, TextList]:
        if isinstance(index, slice):
            return TextList(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: TextList) -> TextList:
        return TextList(super().__add__(other))

    def __repr__(self):
        texts = self[:60]
        texts_str = ",\n ".join([repr(text) for text in texts])
        if len(self) > 60:
            texts_str += ",\n ..."
        return f"TextList([\n {texts_str}]\n)"


class TextTokenList(tuple[TokenList, ...]):
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
    def flat(self) -> TokenList:
        """Flatten the TextTokenList into a TokenList.

        Returns:
            TokenList: The flattened TokenList.
        """
        return TokenList(chain(*self))

    @overload
    def __getitem__(self, index: int) -> TokenList: ...

    @overload
    def __getitem__(self, index: slice) -> TextTokenList: ...

    def __getitem__(self, index: int | slice) -> Union[TokenList, TextTokenList]:
        if isinstance(index, slice):
            return TextTokenList(super().__getitem__(index))
        return super().__getitem__(index)

    def __add__(self, other: TextTokenList) -> TextTokenList:
        return TextTokenList(super().__add__(other))

    def __repr__(self):
        text_tokens = self[:60]
        text_tokens_str = ",\n ".join([tokens._sub_repr() for tokens in text_tokens])
        if len(self) > 60:
            text_tokens_str += ",\n ..."
        return f"TextTokenList([\n {text_tokens_str}]\n)"
