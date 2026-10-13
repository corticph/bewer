"""English config — identical to base."""

from __future__ import annotations

from bewer.configs.config import get_config
from bewer.registry import REGISTRY

__all__: list[str] = []


REGISTRY.configs.register("en", get_config("base"))
