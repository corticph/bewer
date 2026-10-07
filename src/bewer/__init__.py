from importlib.metadata import PackageNotFoundError, version

from bewer import core as core
from bewer import extractors as extractors
from bewer import metrics as metrics
from bewer import preprocessing as preprocessing
from bewer import profiles as profiles
from bewer import reporting as reporting
from bewer.config import BewerConfig, PipelineStep
from bewer.core.dataset import Dataset, DatasetFrozenError
from bewer.core.vocabulary import ExtractorFn, Vocabulary, VocabularyExtractorError, VocabularyFrozenError
from bewer.preprocessing.context import set_pipeline
from bewer.registry import REGISTRY

try:
    __version__ = version("bewer")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Dataset",
    "DatasetFrozenError",
    "Vocabulary",
    "VocabularyExtractorError",
    "VocabularyFrozenError",
    "ExtractorFn",
    "BewerConfig",
    "PipelineStep",
    "REGISTRY",
    "set_pipeline",
    "core",
    "extractors",
    "metrics",
    "preprocessing",
    "profiles",
    "reporting",
    "__version__",
]
