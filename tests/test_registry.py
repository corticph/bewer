"""Tests for the unified registry (bewer.registry)."""

import pytest

from bewer.registry import REGISTRY, ComponentRegistry


class TestComponentRegistry:
    """Tests for the ComponentRegistry class."""

    def test_register_direct_call(self):
        """Direct registration stores the component under the given name."""
        reg = ComponentRegistry("test")
        reg.register("foo", lambda x: x)
        assert "foo" in reg
        assert reg.get("foo")(42) == 42

    def test_register_decorator_named(self):
        """@register("name") uses the explicit name."""
        reg = ComponentRegistry("test")

        @reg.register("explicit")
        def my_func(x):
            return x

        assert "explicit" in reg
        assert "my_func" not in reg

    def test_register_decorator_with_extends(self):
        """@register("name", extends="base") sets the extends attribute."""
        reg = ComponentRegistry("test")

        @reg.register("da", extends="base")
        def my_func():
            return None

        assert reg.get("da").extends == "base"

    def test_register_duplicate_raises(self):
        """Registering a duplicate name raises ValueError."""
        reg = ComponentRegistry("test")
        reg.register("foo", lambda: 1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("foo", lambda: 2)

    def test_get_missing_raises(self):
        """get() raises KeyError for unknown names."""
        reg = ComponentRegistry("test")
        reg.register("alpha", lambda: 1)
        with pytest.raises(KeyError, match="test 'beta' not found"):
            reg.get("beta")

    def test_get_missing_lists_available(self):
        """The error message lists available names."""
        reg = ComponentRegistry("test")
        reg.register("alpha", lambda: 1)
        reg.register("bravo", lambda: 2)
        with pytest.raises(KeyError, match="alpha"):
            reg.get("zzz")

    def test_list_returns_sorted(self):
        """list() returns sorted component names."""
        reg = ComponentRegistry("test")
        reg.register("charlie", lambda: 1)
        reg.register("alpha", lambda: 2)
        reg.register("bravo", lambda: 3)
        assert reg.list() == ["alpha", "bravo", "charlie"]

    def test_contains(self):
        """__contains__ checks membership."""
        reg = ComponentRegistry("test")
        reg.register("foo", lambda: 1)
        assert "foo" in reg
        assert "bar" not in reg


class TestRegistry:
    """Tests for the top-level Registry singleton."""

    def test_configs_has_preregistered(self):
        """The REGISTRY singleton has pre-registered configs."""
        assert "base" in REGISTRY.configs
        assert "en" in REGISTRY.configs
        assert "da" in REGISTRY.configs

    def test_metrics_has_preregistered(self):
        """The REGISTRY singleton exposes the metric registry."""
        assert "wer" in REGISTRY.metrics
        assert "cer" in REGISTRY.metrics

    def test_isolated_restores_state(self, isolated_registry):
        """isolated() restores the registry state after exit."""
        before = set(isolated_registry.configs.list())
        with REGISTRY.isolated():
            REGISTRY.configs.register("temp", lambda: None)
            assert "temp" in REGISTRY.configs
        assert "temp" not in REGISTRY.configs
        assert set(REGISTRY.configs.list()) == before

    def test_isolated_preserves_preregistered(self, isolated_registry):
        """Pre-registered components are still accessible inside isolated()."""
        assert "base" in isolated_registry.configs
        with REGISTRY.isolated():
            assert "base" in REGISTRY.configs
        assert "base" in REGISTRY.configs
