from importlib.metadata import PackageNotFoundError, version

from bewer import configs as configs
from bewer import core as core
from bewer import extractors as extractors
from bewer import metrics as metrics
from bewer import preprocessing as preprocessing
from bewer import reporting as reporting
from bewer.configs.config import BewerConfig, ConfigFrozenError, Transform, get_config
from bewer.core.dataset import Dataset, DatasetFrozenError
from bewer.core.vocabulary import ExtractorFn, Vocabulary, VocabularyExtractorError, VocabularyFrozenError
from bewer.preprocessing.context import set_pipeline
from bewer.registry import CONFIG_REGISTRY

try:
    __version__ = version("bewer")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Dataset",
    "DatasetFrozenError",
    "ConfigFrozenError",
    "Vocabulary",
    "VocabularyExtractorError",
    "VocabularyFrozenError",
    "ExtractorFn",
    "BewerConfig",
    "Transform",
    "get_config",
    "CONFIG_REGISTRY",
    "set_pipeline",
    "core",
    "extractors",
    "metrics",
    "preprocessing",
    "reporting",
    "__version__",
]
