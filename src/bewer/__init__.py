from importlib.metadata import PackageNotFoundError, version

from bewer import core as core
from bewer import metrics as metrics
from bewer import preprocessing as preprocessing
from bewer.core.dataset import Dataset  # noqa: F401

try:
    __version__ = version("bewer")
except PackageNotFoundError:
    __version__ = "0.0.0"

