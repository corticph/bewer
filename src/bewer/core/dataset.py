import ast
from functools import cached_property
from importlib import resources
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

import pandas as pd
from omegaconf import OmegaConf

from bewer.configs.resolve import resolve_pipelines
from bewer.core.example import Example
from bewer.core.key_term import KeyTerm, get_key_term_trie
from bewer.core.text import TokenList
from bewer.metrics.base import MetricCollection

__all__ = ["Dataset", "TextList", "TextTokenList"]

if TYPE_CHECKING:
    from bewer.core.key_term import KeyTermTrie
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

    def __init__(self, config: str | None = None):
        """Initialize the Dataset.

        The dataset must be populated using one of the load_* methods or manually using the add() method.

        Args:
            config (str | None): Path to the configuration file. If None, uses the default configuration.
        """
        self.config_path = self.get_config_path(config)
        self.config = OmegaConf.load(self.config_path)
        self._pipelines = resolve_pipelines(self.config)
        self.examples = []
        self._dynamic_key_term_vocabs = {}
        self._static_key_term_vocabs = {}
        self._cache_key_term_tries = {}
        self.metrics = MetricCollection(self)

    @property
    def pipelines(self):
        return self._pipelines

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

    def add(self, ref: str, hyp: str, key_terms: dict[str, list[str]] | None = None) -> None:
        """Add an example to the dataset."""
        if key_terms is not None:
            key_terms = {name: set(kt_list) for name, kt_list in key_terms.items()}
            for name, kt_set in key_terms.items():
                self._update_static_key_term_vocab(name, kt_set)
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

        # Add examples to the dataset
        for row in df.itertuples(index=False):
            hyp = getattr(row, hyp_col)
            ref = getattr(row, ref_col)
            if len(key_term_cols) > 0:
                key_terms = {}
                for col in key_term_cols:
                    key_terms[col] = getattr(row, col)
            else:
                key_terms = None
            self.add(ref, hyp, key_terms=key_terms)

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

    def add_key_term_list(self, name: str, key_terms: Iterable[str]) -> None:
        """Add a named key term vocabulary to the dataset.

        Key terms are matched against the reference text of each example (including already added examples).

        Args:
            name (str): The name of the key term vocabulary.
            key_terms (Iterable[str]): The key terms to add.
        """
        if not isinstance(key_terms, Iterable) or isinstance(key_terms, str):
            raise TypeError("key_terms must be an iterable of strings")

        key_terms = set(key_terms)

        for key_term in key_terms:
            if not isinstance(key_term, str):
                raise TypeError(f"key_terms must be an iterable of strings, but got element of type {type(key_term)}")

        self._update_dynamic_key_term_vocab(name, key_terms)

    def add_key_term_file(self, name: str, key_term_file: str) -> None:
        """Add a named key term vocabulary to the dataset from a file.

        The file should be plain text and contain one key term per line.

        Key terms are matched against the reference text of each example (including already added examples).

        Args:
            name (str): The name of the key term vocabulary.
            key_term_file (str): Path to the key term file.
        """
        if not Path(key_term_file).is_file():
            raise FileNotFoundError(f"Key term file {key_term_file} not found")

        with open(key_term_file, "r") as f:
            key_terms = f.read().strip().splitlines()

        self.add_key_term_list(name, key_terms)

    def _get_key_term_trie(
        self, vocab: str, normalized: bool = True, add_capitalized: bool = False
    ) -> Optional["KeyTermTrie"]:
        """Get a trie for the specified key term vocabulary."""
        return get_key_term_trie(
            self._dynamic_key_term_vocabs,
            self._cache_key_term_tries,
            vocab,
            normalized=normalized,
            add_capitalized=add_capitalized,
        )

    @staticmethod
    def get_config_path(config_path: str | None) -> str:
        """Get the configuration path."""
        if config_path is None or not Path(config_path).is_file():
            config_path = "base" if config_path is None else config_path
            return resources.files("bewer.configs").joinpath(f"{config_path}.yml")
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

    def _update_dynamic_key_term_vocab(self, name: str, key_terms: set[str]) -> None:
        """Update the dynamic key term vocabulary with new key terms."""
        key_terms = set(KeyTerm(key_term, src=self) for key_term in key_terms)
        if name in self._dynamic_key_term_vocabs:
            self._dynamic_key_term_vocabs[name].update(key_terms)
        else:
            self._dynamic_key_term_vocabs[name] = set(key_terms)

    def _update_static_key_term_vocab(self, name: str, key_terms: set[str]) -> None:
        """Update the static key term vocabulary with new key terms."""
        key_terms = set(KeyTerm(key_term, src=self) for key_term in key_terms)
        if name in self._static_key_term_vocabs:
            self._static_key_term_vocabs[name].update(key_terms)
        else:
            self._static_key_term_vocabs[name] = set(key_terms)

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
