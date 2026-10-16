"""English config."""

from __future__ import annotations

from bewer.configs.config import get_config
from bewer.registry import CONFIG_REGISTRY

__all__: list[str] = []


CONFIG_REGISTRY.register("en", get_config("base"))
