from typing import TYPE_CHECKING, Optional

from bewer.core.text import Text, TextType
from bewer.metrics.base import ExampleMetricCollection

if TYPE_CHECKING:
    from bewer.configs.resolve import Pipelines
    from bewer.core.dataset import Dataset

__all__ = ["Example"]


class Example:
    """
    BeWER example representation.

    Attributes:
        ref (Text): Reference text object.
        hyp (Text): Hypothesis text object.
        metrics (ExampleMetricCollection): Metrics collection for this example.
    """

    def __init__(
        self,
        ref: str,
        hyp: str,
        *,
        pipelines: "Pipelines",
        src: Optional["Dataset"] = None,
        index: Optional[int] = None,
    ):
        """
        Initialize the Example object.

        Args:
            ref: Reference text.
            hyp: Hypothesis text.
            pipelines: The resolved pipeline registry, forwarded to the Text objects (required).
            src: Optional parent Dataset object. Read by the metrics layer and key term vocabulary lookup.
            index: The index of the example in the dataset.
        """
        self._index = index

        self._src = src
        self._pipelines = pipelines

        self.metrics = ExampleMetricCollection(self)
        self.ref = Text(ref, pipelines=self._pipelines, src=self, text_type=TextType.REF)
        self.hyp = Text(hyp, pipelines=self._pipelines, src=self, text_type=TextType.HYP)

    @property
    def index(self) -> Optional[int]:
        """Get the example index."""
        return self._index

    @property
    def src(self) -> Optional["Dataset"]:
        """Get the parent Dataset object, if any."""
        return self._src

    @property
    def pipelines(self):
        return self._pipelines

    @property
    def vocabs(self) -> set[str]:
        """Get the set of all key term vocabularies associated with this example."""
        if self._src is None:
            return set()
        return set(self._src._vocabularies.keys())

    def __hash__(self):
        return hash((self.ref, self.hyp, self._index))

    def __repr__(self):
        ref = self.ref.raw if len(self.ref.raw) <= 45 else self.ref.raw[:42] + "..."
        hyp = self.hyp.raw if len(self.hyp.raw) <= 45 else self.hyp.raw[:42] + "..."
        return f'Example(ref="{ref}", hyp="{hyp}")'
