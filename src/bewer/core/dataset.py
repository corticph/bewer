import ast
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
from bewer.metrics.base import MetricCollection

if TYPE_CHECKING:
    from bewer.core.text import Text


def is_list_literal(s):
    try:
        return isinstance(ast.literal_eval(s), list)
    except (ValueError, SyntaxError):
        return False


class Dataset(object):
    """BeWER dataset.

    Attributes:
        config (...): ...
        examples (list[Example]): A list of Example objects.
        metrics (MetricCollection): A metrics collection for the dataset.
        refs (TextList): The reference texts in the dataset.
        hyps (TextList): The hypothesis texts in the dataset.
    """

    def __init__(self, config: str | None = None):
        """Initialize the Dataset.

        The dataset must be populated using one of the load_* methods or manually using the add() method.

        Args:
            config_path (str | None): Path to the configuration file. If None, uses the default configuration.
        """
        self.config_path = self.get_config_path(config)
        self.config = OmegaConf.load(self.config_path)
        self.pipelines = resolve_pipelines(self.config)
        self.examples = []
        self.vocabs = {}
        self.metrics = MetricCollection(self)

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

    def add(self, ref: str, hyp: str, keywords: dict[str, list[str]] | None = None) -> None:
        """Add an example to the dataset."""
        example = Example(ref, hyp, keywords=keywords, src_dataset=self, index=len(self))
        self.examples.append(example)

    def load_dataset(self, dataset, ref_col="ref", hyp_col="hyp", keyword_cols: list = []) -> None:
        """Load a Hugging Face dataset."""
        raise NotImplementedError("load_dataset() method not implemented.")

    def load_pandas(self, df, ref_col="ref", hyp_col="hyp", keyword_cols: list = []) -> None:
        """Add a pandas DataFrame to the dataset."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        # Prepare and add vocabulary phrases to the tokenizer
        for col in keyword_cols:
            df[col] = self._infer_keyword_column(df[col])

        # Add examples to the dataset
        for row in df.itertuples(index=False):
            hyp = getattr(row, hyp_col)
            ref = getattr(row, ref_col)
            if len(keyword_cols) > 0:
                keywords = {}
                for col in keyword_cols:
                    keywords[col] = getattr(row, col)
            self.add(ref, hyp, keywords=keywords)
        return self

    def load_csv(self, csv_file: str, ref_col="ref", hyp_col="hyp", keyword_cols: list = [], **kwargs) -> None:
        """Add a CSV file to the dataset."""
        df = pd.read_csv(csv_file, **kwargs)
        self.load_pandas(df, ref_col, hyp_col, keyword_cols)
        return self

    @staticmethod
    def get_config_path(config_path: str | None) -> str:
        """Get the configuration path."""
        if config_path is None or not Path(config_path).is_file():
            config_path = "base" if config_path is None else config_path
            with resources.path("bewer.configs", f"{config_path}.yml") as config_path:
                return config_path
        return Path(config_path).resolve()

    def _infer_keyword_column(self, series: pd.Series) -> set:
        """Infer the keyword terms from a pandas Series."""
        if series.map(is_list_literal).all():
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


class TextList(list["Text"]):
    """A list of Text or TokenList objects."""

    @cached_property
    def raw(self) -> list[str]:
        """Get the raw texts as a regular Python list.

        Returns:
            list[str]: The raw text.
        """
        return [text.raw for text in self]

    @cached_property
    def standardized(self) -> list[str]:
        """Get the standardized texts as a regular Python list.

        Returns:
            list[str]: The standardized texts.
        """
        return [text.standardized for text in self]

    @cached_property
    def tokens(self) -> "TextTokenList":
        """Get the tokens as a TextTokenList object.

        Returns:
            TokenList: The tokens.
        """
        return TextTokenList([text.tokens for text in self])

    def __getitem__(self, index: int) -> Union["Text", "TextList"]:
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


class TextTokenList(list["TokenList"]):
    """A list of Text or TokenList objects."""

    @cached_property
    def raw(self) -> list[list[str]]:
        """Get the raw tokens as a regular Python list.

        Returns:
            list[str]: The raw tokens.
        """
        return [tokens.raw for tokens in self]

    @cached_property
    def normalized(self) -> list[str]:
        """Get the normalized tokens as a regular Python list.

        Returns:
            list[str]: The normalized tokens.
        """
        return [tokens.normalized for tokens in self]

    @cached_property
    def flat(self) -> "TokenList":
        """Flatten the TextTokenList into a TokenList.

        Returns:
            TokenList: The flattened TokenList.
        """
        return TokenList(chain(*self))

    def __getitem__(self, index: int | slice) -> Union[TokenList, "TextTokenList"]:
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
