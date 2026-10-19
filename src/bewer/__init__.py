from importlib.metadata import PackageNotFoundError, version

from bewer import configs as configs
from bewer import core as core
from bewer import key_term as key_term
from bewer import metrics as metrics
from bewer import preprocessing as preprocessing
from bewer import reporting as reporting
from bewer.configs import CONFIG_REGISTRY
from bewer.configs.config import BewerConfig, ConfigFrozenError, get_config
from bewer.core.dataset import Dataset, DatasetFrozenError
from bewer.key_term.vocabulary import ExtractorFn, Vocabulary, VocabularyExtractorError, VocabularyFrozenError
from bewer.preprocessing.context import set_pipeline
from bewer.preprocessing.normalization import Transform

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
    "key_term",
    "metrics",
    "preprocessing",
    "reporting",
    "__version__",
]
