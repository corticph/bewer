from importlib.metadata import PackageNotFoundError, version

from bewer import core as core
from bewer import metrics as metrics
from bewer import preprocessing as preprocessing
from bewer import reporting as reporting
from bewer.core.dataset import Dataset
from bewer.core.key_term import KeyTermNotFoundWarning

try:
    __version__ = version("bewer")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Dataset",
    "KeyTermNotFoundWarning",
    "core",
    "metrics",
    "preprocessing",
    "reporting",
    "__version__",
]
