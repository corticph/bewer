"""English config."""

from __future__ import annotations

from bewer.configs.config import CONFIG_REGISTRY, get_config

__all__: list[str] = []


CONFIG_REGISTRY.register("en", get_config("base"))
