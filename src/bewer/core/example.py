import warnings
from typing import TYPE_CHECKING, Iterable, Optional

from bewer.core.keyword import Keyword
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
        keywords (dict[str, set[Keyword]]): Keywords grouped by vocabulary name.
        metrics (ExampleMetricCollection): Metrics collection for this example.
    """

    def __init__(
        self,
        ref: str,
        hyp: str,
        keywords: dict[str, list[str]] | None = None,
        src: Optional["Dataset"] = None,
        index: Optional[int] = None,
    ):
        """
        Initialize the Example object.

        Args:
            ref: Reference text.
            hyp: Hypothesis text.
            keywords: Keywords associated with the example. The keywords are expected
                to be present in the reference text. If not, a warning will be issued and the term will be discarded.
            src: Parent Dataset object. Can be set later via set_source().
            index: The index of the example in the dataset.
        """
        self._index = index

        self._src = None
        if src is not None:
            self.set_source(src)

        self.metrics = ExampleMetricCollection(self)
        self.ref = Text(ref, src=self, text_type=TextType.REF)
        self.hyp = Text(hyp, src=self, text_type=TextType.HYP)
        self.keywords = {}
        self._prepare_and_validate_keywords(keywords, raise_warning=True)
        if self._src is not None:
            self._prepare_and_validate_keywords(self._src._dynamic_keyword_vocabs, raise_warning=False)

    @property
    def index(self) -> Optional[int]:
        """Get the example index."""
        return self._index

    @property
    def src(self) -> Optional["Dataset"]:
        """Get the parent Dataset object."""
        return self._src

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

    def _prepare_and_validate_keywords(
        self,
        keywords: dict[str, Iterable[str]] | None,
        raise_warning: bool = True,
    ) -> None:
        """Prepare keywords dictionary by converting keywords to Text objects."""
        if keywords is None:
            return

        for vocab_name, vocab_keywords in keywords.items():
            validated_keywords = []
            for keyword in vocab_keywords:
                # Check if keyword is present in reference text (case-insensitive)
                if keyword.lower() not in self.ref.raw.lower():
                    if raise_warning:
                        warnings.warn(
                            f"Keyword '{keyword}' not found: Example {self._index}. Will not be included.",
                            KeywordNotFoundWarning,
                        )
                    continue

                # Convert keyword to Keyword object and check if it has valid spans in the reference text
                keyword = Keyword(keyword, src=self)
                if len(keyword.find_in_ref()) == 0:
                    if raise_warning:
                        warnings.warn(
                            f"Keyword '{keyword.raw}' not found in reference text after tokenization: "
                            f"Example {self._index}. Will not be included.",
                            KeywordNotFoundWarning,
                        )
                    continue

                validated_keywords.append(keyword)

            if len(validated_keywords) > 0:
                if vocab_name in self.keywords:
                    self.keywords[vocab_name].update(validated_keywords)
                else:
                    self.keywords[vocab_name] = set(validated_keywords)

    def __hash__(self):
        return hash((self.ref, self.hyp, self._index))

    def __repr__(self):
        ref = self.ref.raw if len(self.ref.raw) <= 45 else self.ref.raw[:42] + "..."
        hyp = self.hyp.raw if len(self.hyp.raw) <= 45 else self.hyp.raw[:42] + "..."
        return f'Example(ref="{ref}", hyp="{hyp}")'
