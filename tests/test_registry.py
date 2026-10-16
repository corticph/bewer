"""Tests for the config registry (bewer.configs.config)."""

import pytest

from bewer.configs import BewerConfig
from bewer.configs.config import CONFIG_REGISTRY, ConfigRegistry


class TestConfigRegistry:
    """Tests for the ConfigRegistry class."""

    def test_register_direct_call(self):
        """Direct registration stores the component under the given name."""
        reg = ConfigRegistry()
        reg.register("foo", lambda x: x)
        assert "foo" in reg
        assert reg.get("foo")(42) == 42

    def test_register_duplicate_raises(self):
        """Registering a duplicate name raises ValueError."""
        reg = ConfigRegistry()
        reg.register("foo", lambda: 1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("foo", lambda: 2)

    def test_get_missing_raises(self):
        """get() raises KeyError for unknown names."""
        reg = ConfigRegistry()
        reg.register("alpha", lambda: 1)
        with pytest.raises(KeyError, match="Config 'beta' not found"):
            reg.get("beta")

    def test_get_missing_lists_available(self):
        """The error message lists available names."""
        reg = ConfigRegistry()
        reg.register("alpha", lambda: 1)
        reg.register("bravo", lambda: 2)
        with pytest.raises(KeyError, match="alpha"):
            reg.get("zzz")

    def test_list_returns_sorted(self):
        """list() returns sorted component names."""
        reg = ConfigRegistry()
        reg.register("charlie", lambda: 1)
        reg.register("alpha", lambda: 2)
        reg.register("bravo", lambda: 3)
        assert reg.list() == ["alpha", "bravo", "charlie"]

    def test_contains(self):
        """__contains__ checks membership."""
        reg = ConfigRegistry()
        reg.register("foo", lambda: 1)
        assert "foo" in reg
        assert "bar" not in reg


class TestConfigRegistrySingleton:
    """Tests for the CONFIG_REGISTRY singleton."""

    def test_has_preregistered(self):
        """The CONFIG_REGISTRY singleton has pre-registered configs."""
        assert "base" in CONFIG_REGISTRY
        assert "en" in CONFIG_REGISTRY
        assert "da" in CONFIG_REGISTRY

    def test_isolated_restores_state(self, isolated_registry):
        """isolated() restores the registry state after exit."""
        before = set(isolated_registry.list())
        with CONFIG_REGISTRY.isolated():
            CONFIG_REGISTRY.register("temp", BewerConfig())
            assert "temp" in CONFIG_REGISTRY
        assert "temp" not in CONFIG_REGISTRY
        assert set(CONFIG_REGISTRY.list()) == before

    def test_isolated_preserves_preregistered(self, isolated_registry):
        """Pre-registered components are still accessible inside isolated()."""
        assert "base" in isolated_registry
        with CONFIG_REGISTRY.isolated():
            assert "base" in CONFIG_REGISTRY
        assert "base" in CONFIG_REGISTRY
