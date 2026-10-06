# Unified Registry & Python Configuration — Implementation Plan

## Goal

Replace the YAML/OmegaConf configuration system with a Python-native configuration
dataclass, and unify registration of all pluggable components (metrics, preprocessing
functions, vocabularies, extractors) under a single registry with a consistent decorator
API.

## Current State (Summary)

| Component    | Registration            | Discovery                  | Validation                |
|--------------|-------------------------|----------------------------|---------------------------|
| Metrics      | `@METRIC_REGISTRY.register("name", **pipeline_defaults)` | Explicit imports in `__init__.py` | Class check at registration; param schema at creation; pipeline names not validated until computation |
| Preprocessing| None — dotted string paths in YAML | `importlib.import_module` at config load | `inspect.signature` param check at load; no existence check until import |
| Vocabularies | Per-dataset dict via `Dataset.add_vocabulary()` | Manual | Type check, name uniqueness per dataset |
| Extractors   | None — plain callables   | Explicit imports           | `callable()` check only   |

---

## Architecture

### 1. Unified Registry (`src/bewer/registry.py`)

A single `Registry` class with typed namespaces for each component kind.

```python
class _ComponentRegistry:
    """Typed namespace for a single component kind."""

    def __init__(self, kind: str):
        self._kind = kind
        self._components: dict[str, Any] = {}

    def register(self, name: str, component, *, allow_override: bool = False):
        if not allow_override and name in self._components:
            raise ValueError(f"{self._kind} '{name}' is already registered")
        self._components[name] = component
        return component

    def get(self, name: str) -> Any:
        try:
            return self._components[name]
        except KeyError:
            raise KeyError(
                f"{self._kind} '{name}' not found. "
                f"Available: {sorted(self._components)}"
            )

    def list(self) -> list[str]:
        return sorted(self._components)

    def __contains__(self, name: str) -> bool:
        return name in self._components


class Registry:
    """Unified registry for all pluggable Bewer components."""

    def __init__(self):
        self.normalizers = _ComponentRegistry("normalizer")
        self.standardizers = _ComponentRegistry("standardizer")
        self.tokenizers = _ComponentRegistry("tokenizer")
        self.metrics = _ComponentRegistry("metric")
        self.extractors = _ComponentRegistry("extractor")
        self.vocabularies = _ComponentRegistry("vocabulary")

    # Convenience decorator factories (one per kind)
    def register_normalizer(self, name: str, **kwargs):
        ...

    def register_tokenizer(self, name: str, **kwargs):
        ...

    def register_standardizer(self, name: str, **kwargs):
        ...

    def register_metric(self, name: str, **pipeline_defaults):
        ...  # delegates to existing MetricRegistry logic

    def register_extractor(self, name: str):
        ...

    def register_vocabulary(self, name: str):
        ...


REGISTRY = Registry()
```

**Key design points:**

- `REGISTRY` is the singleton, replacing `METRIC_REGISTRY`.
- Each namespace is a `_ComponentRegistry` with `register()`, `get()`, `list()`,
  `__contains__`.
- Metric registration retains its existing richer metadata (`pipeline_defaults`,
  `param_defaults`, `param_schema`) — the metric namespace wraps the current
  `MetricRegistry` logic internally.
- Decorator usage:

```python
@REGISTRY.register_normalizer("lowercase")
def lowercase(text: str) -> str:
    return text.lower()

@REGISTRY.register_tokenizer("whitespace")
def whitespace(split_on: str = "") -> re.Pattern:
    ...

@REGISTRY.register_metric("ktr", tokenizer="key_term")
class KTR(Metric):
    ...

@REGISTRY.register_extractor("orthographically_complex")
class OrthographicallyComplexTermExtractor(RegexExtractor):
    ...

@REGISTRY.register_vocabulary("medical")
class MedicalVocabulary:
    terms = ["diabetes", "hypertension", ...]
```

### 2. Component Protocols

Formalize the interfaces that are currently implicit.

```python
# src/bewer/registry.py (or separate protocols.py)

from typing import Protocol, runtime_checkable

@runtime_checkable
class StandardizerFn(Protocol):
    def __call__(self, text: str) -> str: ...

@runtime_checkable
class TokenizerFn(Protocol):
    """Factory that returns a compiled regex pattern."""
    def __call__(self, **kwargs) -> re.Pattern: ...

@runtime_checkable
class NormalizerFn(Protocol):
    def __call__(self, text: str) -> str: ...

@runtime_checkable
class ExtractorFn(Protocol):
    def __call__(self, dataset: "Dataset") -> Iterable[str]: ...
```

These replace the current informal "duck typed" function contracts. Registration
validates against these protocols (with clear error messages).

### 3. Vocabulary Definition vs. Resolution

Split `Vocabulary` into two layers:

- **`VocabularyDefinition`** (registry-level, reusable): declares term sources
  (static terms, files, extractors by name or callable). Stored in the registry.
- **`Vocabulary`** (per-dataset, resolved): the current class, instantiated from a
  definition + a dataset. Holds the Aho-Corasick trie and resolution cache.

```python
@dataclass(frozen=True)
class VocabularyDefinition:
    name: str
    terms: frozenset[str] = frozenset()
    files: tuple[str, ...] = ()
    extractors: tuple[str, ...] = ()  # registered extractor names

    def resolve(self, dataset) -> "Vocabulary":
        vocab = Vocabulary(self.name)
        if self.terms:
            vocab.add_terms(self.terms)
        for f in self.files:
            vocab.add_file(f)
        for ext_name in self.extractors:
            vocab.add_extractor(REGISTRY.extractors.get(ext_name))
        return vocab
```

**Registered vocabulary** (word-list style):

```python
@REGISTRY.register_vocabulary("medical")
class MedicalVocabulary(VocabularyDefinition):
    name = "medical"
    terms = frozenset({"diabetes", "hypertension", "insulin", ...})
```

**Builder for one-off vocabularies** (unchanged API, returns a `VocabularyDefinition`
that can be registered or used directly):

```python
vocab = (
    VocabularyDefinition("custom")
    .add_terms(["diabetes", "blood sugar"])
    .add_extractor("medical_acronyms")
)
dataset.add_vocabulary(vocab)
```

**Auto-resolution**: When a metric references `vocab="medical"`, the dataset checks
`_vocabularies` first, then falls back to `REGISTRY.vocabularies.get("medical")` and
resolves it. No explicit `add_vocabulary` needed for registered vocabularies.

### 4. BewerConfig Dataclass

```python
@dataclass
class PipelineStep:
    """A single step in a preprocessing pipeline."""
    component: str | callable  # registered name or direct callable
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BewerConfig:
    standardizers: dict[str, list[PipelineStep | str | callable]] = field(
        default_factory=dict
    )
    tokenizers: dict[str, PipelineStep | str | callable] = field(
        default_factory=dict
    )
    normalizers: dict[str, list[PipelineStep | str | callable]] = field(
        default_factory=dict
    )
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    vocabularies: dict[str, str | VocabularyDefinition] = field(default_factory=dict)

    # Language/domain profile reference
    profile: str | None = None

    # User overrides applied after profile
    overrides: dict[str, Any] = field(default_factory=dict)
```

**Config resolution** replaces `resolve.py`:

```python
def resolve_config(config: BewerConfig) -> ResolvedConfig:
    # 1. Start from profile if specified
    if config.profile:
        profile = REGISTRY.profiles.get(config.profile)
        base = profile()
    else:
        base = default_config()

    # 2. Apply user overrides
    merged = merge_configs(base, config)

    # 3. Resolve all components (name -> callable, build Normalizer/Tokenizer)
    standardizers = {
        name: _resolve_standardizer(steps)
        for name, steps in merged.standardizers.items()
    }
    tokenizers = {
        name: _resolve_tokenizer(step)
        for name, step in merged.tokenizers.items()
    }
    normalizers = {
        name: _resolve_normalizer(steps)
        for name, steps in merged.normalizers.items()
    }

    return ResolvedConfig(
        standardizers=standardizers,
        tokenizers=tokenizers,
        normalizers=normalizers,
        metrics=merged.metrics,
    )


def _resolve_step(step):
    """Resolve a PipelineStep to (callable, params)."""
    if isinstance(step, str):
        # Registered name — look up in registry (kind inferred from context)
        return REGISTRY.<kind>.get(step), {}
    elif isinstance(step, PipelineStep):
        if isinstance(step.component, str):
            return REGISTRY.<kind>.get(step.component), step.params
        else:
            return step.component, step.params
    elif callable(step):
        # Direct callable — auto-register under a generated name
        return step, {}
```

**Key improvement**: Pipeline name validation happens at `resolve_config()` time —
typos produce immediate, clear errors listing available names.

### 5. Language/Domain Profiles

Replace `configs/languages/*.yml` with Python modules.

```python
# src/bewer/profiles/__init__.py
# Profiles auto-register via decorator

@REGISTRY.register_profile("base")
def base_config() -> BewerConfig:
    return BewerConfig(
        standardizers={
            "default": [
                PipelineStep("nfc"),
                PipelineStep("normalize_apostrophe_variants"),
                PipelineStep("normalize_hyphen_variants"),
                PipelineStep("normalize_slash_variants"),
            ],
        },
        tokenizers={
            "default": PipelineStep("strip_punctuation_keep_symbols_pattern",
                                    params={"split_on_escaped": "-/"}),
            "with_punctuation": PipelineStep("keep_symbols_and_punctuation_pattern",
                                    params={"punct_chars": '.,!?:;"-/()""«»„¡¿',
                                            "keep_newlines": True}),
            "key_term": PipelineStep("strip_punctuation_keep_symbols_pattern",
                                    params={"split_on_escaped": "-/'"}),
            "orthographically_complex_term": PipelineStep("strip_punctuation_keep_symbols_pattern",
                                    params={"split_on_escaped": "/'"}),
        },
        normalizers={
            "default": [
                PipelineStep("lowercase"),
                PipelineStep("transliterate_latin_letters"),
                PipelineStep("transliterate_symbols"),
            ],
            "cased": [
                PipelineStep("transliterate_latin_letters"),
                PipelineStep("transliterate_symbols"),
            ],
        },
    )


@REGISTRY.register_profile("da", extends="base")
def danish_config() -> BewerConfig:
    base = base_config()
    base.normalizers["default"] = [
        PipelineStep("lowercase"),
        PipelineStep("transliterate_latin_letters", params={"preserve": "æøå"}),
        PipelineStep("transliterate_symbols"),
    ]
    return base


@REGISTRY.register_profile("fr", extends="base")
def french_config() -> BewerConfig:
    base = base_config()
    base.tokenizers["default"] = PipelineStep(
        "strip_punctuation_keep_symbols_pattern",
        params={"split_on_escaped": "-/'"},
    )
    base.normalizers["default"] = [
        PipelineStep("lowercase"),
        PipelineStep("transliterate_latin_letters",
                     params={"preserve": "àâäçéèêëîïôöùûüÿœæ"}),
        PipelineStep("transliterate_symbols"),
    ]
    return base
```

**Domain profiles** stack on language profiles:

```python
@REGISTRY.register_profile("medical_da", extends="da")
def medical_danish_config() -> BewerConfig:
    base = danish_config()
    base.vocabularies = {"medical": "medical"}
    return base
```

**Profile inheritance** is simple function composition — a profile calls its parent
and modifies the result. No MRO, no copy-on-inherit complexity. The `extends` parameter
is purely for documentation and potential dependency-ordering.

### 6. Dataset Changes

```python
class Dataset:
    def __init__(self, config=None, language=None):
        # New: accept BewerConfig, profile name, or None (defaults)
        if isinstance(config, BewerConfig):
            self._config = config
        elif isinstance(config, str):
            # Profile name or path to YAML (backward compat)
            if config in REGISTRY.profiles:
                self._config = REGISTRY.profiles.get(config)()
            else:
                self._config = self._load_yaml_config(config)
        elif config is None and language is not None:
            self._config = REGISTRY.profiles.get(language)()
        else:
            self._config = REGISTRY.profiles.get("base")()

        self._resolved = resolve_config(self._config)
        self._pipelines = self._resolved.pipelines
        self._init_blank_state()

    def add_vocabulary(self, vocab):
        """Accepts VocabularyDefinition or Vocabulary (backward compat)."""
        if isinstance(vocab, VocabularyDefinition):
            vocab = vocab.resolve(self)
        # ... existing logic

    def _get_vocabulary(self, name: str) -> Vocabulary:
        """Resolve a vocabulary by name — registered or attached."""
        if name in self._vocabularies:
            return self._vocabularies[name]
        if name in REGISTRY.vocabularies:
            vocab = REGISTRY.vocabularies.get(name).resolve(self)
            self._register_derived_vocabulary(vocab)
            return vocab
        raise ValueError(
            f"Vocabulary '{name}' not found. "
            f"Attached: {sorted(self._vocabularies)}, "
            f"Registered: {REGISTRY.vocabularies.list()}"
        )
```

**Pipeline name validation** moves to `resolve_config()` — a typo like
`normalizer="defualt"` fails immediately with:
```
ValueError: Normalizer 'defualt' not found. Available: ['cased', 'default']
```

### 7. What Does NOT Change

- **`Pipelines` namedtuple** — same structure, just built differently.
- **ContextVar mechanism** (`context.py`) — unchanged.
- **`pipeline_cached_property`** (`caching.py`) — unchanged.
- **`Text.standardized`, `Text.tokens`, `Token.normalized`** — unchanged.
- **`Metric` base class, `MetricParams`, `metric_value`, `dependency`** — unchanged.
- **`MetricCollection`, `ExampleMetricCollection`** — unchanged in shape.
- **`Vocabulary` resolution cache** (`WeakKeyDictionary`) — unchanged.
- **`KeyTermTrie`** — unchanged.
- **Reporting system** — completely unaffected.
- **`set_pipeline` context manager** — unchanged.

### 8. YAML Round-Trip (Optional)

The registry makes YAML serialization straightforward:

```python
def to_yaml(config: BewerConfig) -> str:
    """Serialize a BewerConfig to YAML. Only registered names round-trip."""
    ...

def from_yaml(yaml_str: str) -> BewerConfig:
    """Parse YAML into a BewerConfig, resolving names via registry."""
    ...
```

A `PipelineStep("lowercase")` serializes as `lowercase:` (name only).
A `PipelineStep("transliterate_latin_letters", {"preserve": "æøå"})` serializes as:
```yaml
transliterate_latin_letters:
  preserve: æøå
```
A direct callable (not registered) raises `SerializationError` — to share, register first.

---

## Implementation Steps

### Phase 1: Registry Core (no breaking changes)

1. **Create `src/bewer/registry.py`** with `Registry`, `_ComponentRegistry`,
   `REGISTRY` singleton, and `Protocol` definitions.

2. **Register all existing preprocessing functions** using the new decorators,
   keeping the functions in their current locations:
   - `normalization.py`: `@REGISTRY.register_normalizer("lowercase")`, etc.
   - `tokenization.py`: `@REGISTRY.register_tokenizer("whitespace")`, etc.
   - These decorators are additive — the functions still work as plain callables.

3. **Register all existing extractors**:
   - `RegexExtractor` — not registered (it's a base class, not a concrete extractor)
   - `OrthographicallyComplexTermExtractor` — `@REGISTRY.register_extractor("orthographically_complex")`

4. **Migrate `MetricRegistry` to use `REGISTRY.metrics`** internally:
   - `MetricRegistry.register_metric()` delegates to `REGISTRY.metrics.register()`
   - `MetricRegistry.resolve_params()` and `create_metric()` unchanged
   - `METRIC_REGISTRY` becomes a thin compat layer pointing at `REGISTRY.metrics`

5. **Add tests** for the registry: registration, lookup, duplicate detection, listing,
   protocol validation.

### Phase 2: BewerConfig & Profiles (breaking change: Dataset init)

6. **Create `src/bewer/config.py`** with `BewerConfig`, `PipelineStep`,
   `ResolvedConfig`, `resolve_config()`.

7. **Create `src/bewer/profiles/__init__.py`** with `base`, `en`, `da`, `de`, `fr`
   profiles, registered via `@REGISTRY.register_profile()`.

8. **Move base pipeline definitions** from `configs/base.yml` to the `base` profile
   function.

9. **Update `Dataset.__init__`** to accept `BewerConfig | str | None`:
   - `BewerConfig` → resolve directly
   - `str` (profile name) → look up in `REGISTRY.profiles`
   - `str` (file path ending in `.yml`) → backward-compat YAML loading
   - `None` → `"base"` profile

10. **Update `Dataset.add_vocabulary`** to accept `VocabularyDefinition` in addition
    to `Vocabulary`.

11. **Add `Dataset._get_vocabulary(name)`** for auto-resolution from registry.

12. **Update metric param validation** (`MetricParams.validate()`) to use
    `Dataset._get_vocabulary(name)` instead of direct `dataset._vocabularies` lookup.

13. **Remove `OmegaConf` dependency** from `Dataset.__init__` (keep for YAML
    backward-compat shim only).

14. **Add tests** for config resolution, profile loading, pipeline name validation,
    vocabulary auto-resolution.

### Phase 3: Vocabulary Definitions

15. **Create `VocabularyDefinition`** class in `vocabulary.py` alongside
    `Vocabulary`.

16. **Register built-in vocabularies** (e.g., orthographically complex terms as a
    vocabulary definition, not just an auto-registered extractor).

17. **Update `orthographically_complex_term.py`** to use
    `REGISTRY.vocabularies.get("orthographically_complex_terms")` instead of
    `_ensure_orthographically_complex_term_vocabulary()`.

18. **Deprecate `Dataset.add_vocabulary()`** — keep working but emit
    `DeprecationWarning` pointing to registry-based auto-resolution.

19. **Add tests** for vocabulary definitions, resolution, auto-resolution.

### Phase 4: Cleanup & Migration

20. **Remove YAML config files** (`configs/base.yml`, `configs/languages/*.yml`).
    Keep `configs/resolve.py` temporarily for the YAML backward-compat shim.

21. **Remove `OmegaConf`** dependency from core (keep only in YAML compat shim).

22. **Remove `_resolve_function()`** and `importlib.import_module` string resolution
    from the main code path.

23. **Update `TODO.md`** — mark "Validate pipeline names" and "Config-based metric
    registration" as resolved.

24. **Update `README.md`** — replace YAML references with Python config examples.

25. **Update `CLAUDE.md`** — replace architecture description.

26. **Remove `flags.py`** constants (`TOKENIZERS`, `STANDARDIZERS`, `NORMALIZERS`,
    `DEFAULT`) — replace with registry lookups or inline constants.

27. **Full test pass** — ensure all existing tests pass (with config migration)
    and new tests cover the registry, config, and profile systems.

### Phase 5: YAML Round-Trip (Optional, Low Priority)

28. **Add `to_yaml()` / `from_yaml()`** in `config.py`.

29. **Add tests** for YAML serialization round-trip.

---

## File Changes Summary

| File | Action |
|------|--------|
| `src/bewer/registry.py` | **New** — unified registry, protocols |
| `src/bewer/config.py` | **New** — `BewerConfig`, `PipelineStep`, `resolve_config()` |
| `src/bewer/profiles/__init__.py` | **New** — language/domain profiles |
| `src/bewer/core/dataset.py` | **Modify** — new config loading, vocabulary auto-resolution |
| `src/bewer/core/vocabulary.py` | **Modify** — add `VocabularyDefinition` |
| `src/bewer/preprocessing/normalization.py` | **Modify** — add `@REGISTRY.register_normalizer` decorators |
| `src/bewer/preprocessing/tokenization.py` | **Modify** — add `@REGISTRY.register_tokenizer` decorators |
| `src/bewer/metrics/base.py` | **Modify** — `MetricRegistry` delegates to `REGISTRY.metrics` |
| `src/bewer/metrics/orthographically_complex_term.py` | **Modify** — use registry vocab resolution |
| `src/bewer/extractors/orthographically_complex_term.py` | **Modify** — add `@REGISTRY.register_extractor` |
| `src/bewer/extractors/regex.py` | Unchanged (base class) |
| `src/bewer/__init__.py` | **Modify** — export `REGISTRY`, `BewerConfig`, `VocabularyDefinition` |
| `src/bewer/configs/base.yml` | **Remove** (Phase 4) |
| `src/bewer/configs/languages/*.yml` | **Remove** (Phase 4) |
| `src/bewer/configs/resolve.py` | **Remove** (Phase 4) |
| `src/bewer/flags.py` | **Remove** (Phase 4) |
| `tests/` | **Modify** — migrate all config fixtures to `BewerConfig` |
| `TODO.md` | **Modify** — mark resolved items |
| `README.md` | **Modify** — update config examples |
| `CLAUDE.md` | **Modify** — update architecture description |

---

## Risk Assessment

**Low risk:**
- Registry is additive — existing code keeps working during Phase 1.
- ContextVar/caching mechanism is completely orthogonal — zero changes.
- Metric registration API is backward compatible (`METRIC_REGISTRY` becomes a
  thin wrapper).

**Medium risk:**
- `Dataset.__init__` signature change (Phase 2) — mitigated by accepting both
  `BewerConfig` and `str` (profile name or YAML path).
- Vocabulary auto-resolution changes the metric creation flow — need to ensure
  `_register_derived_vocabulary` path still works for late registration.

**What could go wrong:**
- Profile inheritance via function composition is simple but not composable for
  complex domain+language stacking. If this grows, consider a profile composition
  protocol (e.g., `profile.compose(other)`).
- `VocabularyDefinition` as a frozen dataclass with builder methods
  (`add_terms` returning new instances) is a pattern change from the mutable
  `Vocabulary` builder. Need to ensure the builder is ergonomic.
- Removing `OmegaConf` entirely requires a YAML parser for the backward-compat
  shim. Can use `pyyaml` directly (already a dependency) instead of `OmegaConf`.

---

## Open Questions

1. **Profile composition**: Should profiles support multiple inheritance (e.g.,
   `medical` + `da`)? Currently the design uses single `extends` via function
   composition. If multi-inheritance is needed, a `compose()` method or
   `@register_profile("medical_da", extends=["medical", "da"])` could work.

2. **VocabularyDefinition builder mutability**: Should `VocabularyDefinition` be
   frozen (immutable, builder returns new instances) or mutable (like current
   `Vocabulary`)? Frozen is safer for registry storage; mutable is more ergonomic
   for one-off construction. Proposal: mutable during construction, frozen on
   registration (like `Vocabulary._freeze()`).

3. **Metric namespace in config**: Should `BewerConfig.metrics` define named metric
   presets (as in TODO.md's `metrics:` section) or just list metric names to
   pre-compute? The registry already handles metric registration; config could
   define presets with fixed params (e.g., `my_wer: {base: wer, normalizer: cased}`).

4. **Backward compat duration**: How long to keep the YAML loading shim? Until
   1.0? Indefinitely?
