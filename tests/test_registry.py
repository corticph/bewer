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

    def test_transforms_has_preregistered(self):
        """The REGISTRY singleton has pre-registered transforms."""
        assert "lowercase" in REGISTRY.transforms
        assert "nfc" in REGISTRY.transforms
        assert "strip_punctuation" in REGISTRY.transforms

    def test_tokenizers_has_preregistered(self):
        """The REGISTRY singleton has pre-registered tokenizers."""
        assert "whitespace_pattern" in REGISTRY.tokenizers
        assert "strip_punctuation_pattern" in REGISTRY.tokenizers

    def test_extractors_has_preregistered(self):
        """The REGISTRY singleton has the orthographically_complex extractor."""
        assert "orthographically_complex" in REGISTRY.extractors

    def test_metrics_has_preregistered(self):
        """The REGISTRY singleton exposes the metric registry."""
        assert "wer" in REGISTRY.metrics
        assert "cer" in REGISTRY.metrics

    def test_isolated_restores_state(self, isolated_registry):
        """isolated() restores the registry state after exit."""
        before = set(isolated_registry.transforms.list())
        with REGISTRY.isolated():
            REGISTRY.transforms.register("temp", lambda x: x)
            assert "temp" in REGISTRY.transforms
        assert "temp" not in REGISTRY.transforms
        assert set(REGISTRY.transforms.list()) == before

    def test_isolated_preserves_preregistered(self, isolated_registry):
        """Pre-registered components are still accessible inside isolated()."""
        assert "lowercase" in isolated_registry.transforms
        with REGISTRY.isolated():
            assert "lowercase" in REGISTRY.transforms
        assert "lowercase" in REGISTRY.transforms


class TestSignatureValidation:
    """Tests for at-registration signature validation."""

    def test_transform_with_no_params_rejected(self, isolated_registry):
        """A transform with zero required positional params is rejected."""
        with pytest.raises(TypeError, match="exactly one required positional"):

            @isolated_registry.transforms.register
            def bad_transform() -> str:
                return ""

    def test_transform_with_no_params_rejected_message(self, isolated_registry):
        """A clear error message is produced."""
        with pytest.raises(TypeError, match="exactly one required positional parameter"):

            @isolated_registry.transforms.register
            def bad_transform() -> str:
                return ""

    def test_transform_with_two_required_positional_rejected(self, isolated_registry):
        """A transform with two required positional params is rejected."""
        with pytest.raises(TypeError, match="exactly one required positional"):

            @isolated_registry.transforms.register
            def bad_transform(text: str, extra: str) -> str:
                return text

    def test_transform_with_optional_params_accepted(self, isolated_registry):
        """A transform with one required + optional keyword params is accepted."""

        @isolated_registry.transforms.register
        def good_transform(text: str, preserve: str = "") -> str:
            return text

        assert "good_transform" in isolated_registry.transforms

    def test_tokenizer_with_positional_only_rejected(self, isolated_registry):
        """A tokenizer with positional-only params is rejected."""
        with pytest.raises(TypeError, match="positional-only"):

            @isolated_registry.tokenizers.register
            def bad_tokenizer(text, /) -> None:
                pass

    def test_tokenizer_with_keyword_params_accepted(self, isolated_registry):
        """A tokenizer with keyword params (with or without defaults) is accepted."""

        @isolated_registry.tokenizers.register
        def good_tokenizer(punct_chars: str, keep_newlines: bool = True) -> None:
            pass

        assert "good_tokenizer" in isolated_registry.tokenizers

    def test_extractor_instance_not_signature_checked(self, isolated_registry):
        """Extractor instances (not functions) bypass signature validation."""
        isolated_registry.extractors.register("my_extractor", object())
        assert "my_extractor" in isolated_registry.extractors
