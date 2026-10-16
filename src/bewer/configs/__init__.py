"""Config presets and configuration system for BeWER.

This package contains ``BewerConfig``, ``Transform``, and related utilities
in ``config.py``, plus per-language config presets registered on import.
"""

from bewer.configs import base as base  # noqa: F401
from bewer.configs import da as da  # noqa: F401
from bewer.configs import de as de  # noqa: F401
from bewer.configs import en as en  # noqa: F401
from bewer.configs import fr as fr  # noqa: F401
from bewer.configs.config import (
    CONFIG_REGISTRY,
    BewerConfig,
    ConfigFrozenError,
    ConfigRegistry,
    Transform,
    get_config,
)

__all__ = [
    "BewerConfig",
    "CONFIG_REGISTRY",
    "ConfigFrozenError",
    "ConfigRegistry",
    "Transform",
    "get_config",
]
