import warnings
from typing import TYPE_CHECKING, Optional

from bewer.core.text import Text, TextType
from bewer.metrics.base import ExampleMetricCollection

if TYPE_CHECKING:
    from bewer.core.dataset import Dataset


class KeywordNotFoundWarning(UserWarning):
    pass


warnings.filterwarnings("always", category=KeywordNotFoundWarning)


class Example:
    """
    BeWER example representation.

    Attributes:
        ref (Text): Reference text object.
        hyp (Text): Hypothesis text object.
    """

    def __init__(
        self,
        ref: str,
        hyp: str,
        keywords: dict[str, list[str]] | None = None,
        src_dataset: Optional["Dataset"] = None,
        index: Optional[int] = None,
    ):
        """
        Initialize the Example object.

        Args:
            ref (str): Reference text.
            hyp (str): Hypothesis text.
            keywords (dict[str, list[str]], optional): Keywords associated with the example. The keywords must be
                present in the reference text, otherwise they will be ignored. Defaults to None.
            src_dataset (Dataset, optional): The source Dataset object. Defaults to None.
            index (int, optional): The index of the example in the dataset. Defaults to None.
        """
        self._src_dataset = src_dataset
        self._index = index

        self.metrics = ExampleMetricCollection(self)
        self.ref = Text(ref, src_example=self, text_type=TextType.REF)
        self.hyp = Text(hyp, src_example=self, text_type=TextType.HYP)
        self.keywords = self._prepare_and_validate_keywords(keywords)

    @property
    def index(self) -> Optional[int]:
        return self._index

    @property
    def src(self) -> Optional["Dataset"]:
        return self._src_dataset

    def _prepare_and_validate_keywords(self, keywords: dict[str, list[str]] | None) -> dict[str, list[str]]:
        """Prepare keywords dictionary by converting terms to Text objects."""
        if keywords is None:
            return {}

        validated_keywords = {}
        for name, terms in keywords.items():
            validated_terms = []
            for term in terms:
                if term.lower() not in self.ref.raw.lower():
                    warnings.warn(
                        f"Keyword '{term}' not found in reference for example {self._index}. Will not be included.",
                        KeywordNotFoundWarning,
                    )
                    continue
                validated_terms.append(Text(term, src_example=self, text_type=TextType.KEYWORD))

            if len(validated_terms) > 0:
                validated_keywords[name] = validated_terms

        return validated_keywords

    def __hash__(self):
        return hash((self.ref, self.hyp, self._index))

    def __repr__(self):
        ref = self.ref.raw if len(self.ref.raw) <= 45 else self.ref.raw[:42] + "..."
        hyp = self.hyp.raw if len(self.hyp.raw) <= 45 else self.hyp.raw[:42] + "..."
        return f'Example(ref="{ref}", hyp="{hyp}")'
