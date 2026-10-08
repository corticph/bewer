"""Tests for the unified registry (bewer.registry)."""

import pytest

from bewer.registry import REGISTRY, ComponentNotFoundError, ComponentRegistry


class TestComponentRegistry:
    """Tests for the ComponentRegistry class."""

    def test_register_direct_call(self):
        """Direct registration stores the component under the given name."""
        reg = ComponentRegistry("test")
        reg.register("foo", lambda x: x)
        assert "foo" in reg
        assert reg.get("foo")(42) == 42

    def test_register_decorator_bare(self):
        """Bare @register defaults the name to fn.__name__."""
        reg = ComponentRegistry("test")

        @reg.register
        def my_func(x):
            return x * 2

        assert "my_func" in reg
        assert reg.get("my_func")(3) == 6

    def test_register_decorator_named(self):
        """@register("name") uses the explicit name."""
        reg = ComponentRegistry("test")

        @reg.register("explicit")
        def my_func(x):
            return x

        assert "explicit" in reg
        assert "my_func" not in reg

    def test_register_decorator_with_attrs(self):
        """@register(attr=True) sets attributes on the function."""
        reg = ComponentRegistry("test")

        @reg.register(custom_attr=True)
        def my_func(x):
            return x

        assert reg.get("my_func").custom_attr is True

    def test_register_duplicate_raises(self):
        """Registering a duplicate name without allow_override raises."""
        reg = ComponentRegistry("test")
        reg.register("foo", lambda: 1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("foo", lambda: 2)

    def test_register_duplicate_allow_override(self):
        """allow_override=True silently replaces the component."""
        reg = ComponentRegistry("test")
        reg.register("foo", lambda: 1)
        reg.register("foo", lambda: 2, allow_override=True)
        assert reg.get("foo")() == 2

    def test_get_missing_raises(self):
        """get() raises ComponentNotFoundError for unknown names."""
        reg = ComponentRegistry("test")
        reg.register("alpha", lambda: 1)
        with pytest.raises(ComponentNotFoundError, match="Test 'beta' not found"):
            reg.get("beta")

    def test_get_missing_did_you_mean(self):
        """The error includes a 'did you mean?' hint for close matches."""
        reg = ComponentRegistry("test")
        reg.register("lowercase", lambda: 1)
        with pytest.raises(ComponentNotFoundError, match="Did you mean 'lowercase'"):
            reg.get("lowercas")

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

    def test_iter(self):
        """__iter__ yields names."""
        reg = ComponentRegistry("test")
        reg.register("a", lambda: 1)
        reg.register("b", lambda: 2)
        assert set(iter(reg)) == {"a", "b"}

    def test_len(self):
        """__len__ returns the count."""
        reg = ComponentRegistry("test")
        assert len(reg) == 0
        reg.register("a", lambda: 1)
        assert len(reg) == 1


class TestRegistry:
    """Tests for the top-level Registry singleton."""

    def test_extractors_has_preregistered(self):
        """The REGISTRY singleton has the orthographically_complex extractor."""
        assert "orthographically_complex" in REGISTRY.extractors

    def test_metrics_has_preregistered(self):
        """The REGISTRY singleton exposes the metric registry."""
        assert "wer" in REGISTRY.metrics
        assert "cer" in REGISTRY.metrics

    def test_isolated_restores_state(self, isolated_registry):
        """isolated() restores the registry state after exit."""
        before = set(isolated_registry.extractors.list())
        with REGISTRY.isolated():
            REGISTRY.extractors.register("temp", object())
            assert "temp" in REGISTRY.extractors
        assert "temp" not in REGISTRY.extractors
        assert set(REGISTRY.extractors.list()) == before

    def test_isolated_preserves_preregistered(self, isolated_registry):
        """Pre-registered components are still accessible inside isolated()."""
        assert "orthographically_complex" in isolated_registry.extractors
        with REGISTRY.isolated():
            assert "orthographically_complex" in REGISTRY.extractors
        assert "orthographically_complex" in REGISTRY.extractors
