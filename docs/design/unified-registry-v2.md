# Unified Registry & Python Configuration — Revised Implementation Plan

> **Revision 2.** Incorporates all findings from the review in
> `unified-registry-review.md`. Each changed section is marked **[revised]** with a
> note on what changed. Dropped sections are retained as strikethrough summaries for
> traceability.

---

## Goal

Replace the YAML/OmegaConf configuration system with a Python-native configuration
dataclass, and unify registration of all pluggable components (preprocessing
transforms, tokenizers, metrics, vocabularies, extractors) under a single registry
with a consistent decorator API.

Two separate namespaces must not be conflated (review §1):

| Namespace   | Examples                                      | Defined by  | Validated when            |
|-------------|-----------------------------------------------|-------------|---------------------------|
| **Component** | `lowercase`, `nfc`, `strip_punctuation_keep_symbols_pattern` | registry | at config resolution |
| **Variant**    | `default`, `cased`, `key_term`                | config      | at metric request (`MetricCollection.get()`) |

## Current State (Summary) [revised]

| Component    | Registration            | Discovery                  | Validation                |
|--------------|-------------------------|----------------------------|---------------------------|
| Metrics      | `@METRIC_REGISTRY.register("name", **pipeline_defaults, **param_defaults)` | Explicit imports in `__init__.py` | Class check at registration; param schema at creation; **variant** names not validated until computation |
| Preprocessing| None — dotted string paths in YAML | `importlib.import_module` at `Dataset.__init__` | `inspect.signature` param check at load; missing functions fail at construction with `ModuleNotFoundError`/`AttributeError` (review §1 correction) |
| Vocabularies | Per-dataset dict via `Dataset.add_vocabulary()` | Manual | Type check, name uniqueness per dataset |
| Extractors   | None — plain callables   | Explicit imports           | `callable()` check only   |

**The existing `Vocabulary` is already dataset-independent** (review §2). Its
docstring says it "holds only its *definition*; the concrete key terms are resolved
lazily per dataset". The per-dataset state lives in the `WeakKeyDictionary` cache. A
single `Vocabulary` can be shared across multiple datasets.

---

## Architecture

### 1. Unified Registry (`src/bewer/registry.py`) [revised: single transforms namespace, extractor instances, profiles namespace, error type, test isolation]

A single `Registry` class with typed namespaces for each component kind.

```python
class ComponentNotFoundError(LookupError):
    """Raised when a registry lookup fails. Includes 'did you mean' hints."""
    def __init__(self, kind: str, name: str, available: list[str]):
        import difflib
        hints = difflib.get_close_matches(name, available, n=3, cutoff=0.6)
        msg = f"{kind.capitalize()} '{name}' not found. Available: {sorted(available)}"
        if hints:
            msg += f". Did you mean: {hints}?"
        super().__init__(msg)


class _ComponentRegistry:
    """Typed namespace for a single component kind."""

    def __init__(self, kind: str):
        self._kind = kind
        self._components: dict[str, Any] = {}

    def register(self, name=None, component=None, *, allow_override: bool = False,
                 **attrs):
        """Register a component. Works as a direct call or a decorator.

        Direct call:
            REGISTRY.transforms.register("lowercase", lowercase_fn, length_preserving=True)
            REGISTRY.extractors.register("oct", OrthographicallyComplexTermExtractor())

        Decorator (name defaults to fn.__name__):
            @REGISTRY.transforms.register
            def lowercase(text): ...

            @REGISTRY.transforms.register("lc", length_preserving=True)
            def lowercase(text): ...

        Extra keyword args (e.g. token_only, length_preserving) are set as
        attributes on the registered function.
        """
        # Decorator form: called as @REGISTRY.transforms.register or
        # @REGISTRY.transforms.register("name", ...)
        if component is None and callable(name):
            component = name
            name = component.__name__

        def _do_register(name, component):
            if not isinstance(name, str):
                raise TypeError(f"Name must be a string, got {type(name)}")
            if not allow_override and name in self._components:
                raise ValueError(f"{self._kind} '{name}' is already registered")
            for k, v in attrs.items():
                setattr(component, k, v)
            self._components[name] = component
            return component

        if component is not None:
            return _do_register(name, component)
        # Return a decorator for @REGISTRY.transforms.register("name", ...)
        def decorator(fn):
            n = name or fn.__name__
            return _do_register(n, fn)
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._components:
            raise ComponentNotFoundError(self._kind, name, list(self._components))
        return self._components[name]

    def list(self) -> list[str]:
        return sorted(self._components)

    def __contains__(self, name: str) -> bool:
        return name in self._components


class Registry:
    """Unified registry for all pluggable Bewer components."""

    def __init__(self):
        # Single namespace for str->str transforms (review §5).
        # Both standardizers and normalizers are the same type; nfc is a normalizer
        # function used as a standardizer. Metadata (token_only, length_preserving)
        # is set at registration, not via _set_attrs.
        self.transforms = _ComponentRegistry("transform")
        self.tokenizers = _ComponentRegistry("tokenizer")
        self.metrics = None          # set to METRIC_REGISTRY after import (review §10)
        self.extractors = _ComponentRegistry("extractor")
        self.vocabularies = _ComponentRegistry("vocabulary")
        self.profiles = _ComponentRegistry("profile")

    # All namespaces share the same _ComponentRegistry.register API.
    # register() works as both a direct call and a decorator:
    #
    #   REGISTRY.transforms.register("lowercase", lowercase_fn, length_preserving=True)
    #   @REGISTRY.transforms.register  # name defaults to fn.__name__
    #   def nfc(text): ...
    #
    # Metadata (token_only, length_preserving) for transforms is passed as
    # keyword args to register() and set as attributes on the function.

    # --- profiles: callables that return BewerConfig ---
    # Profiles store (fn, extends) tuples. The registry builds the chain.
    def register_profile(self, name: str, *, extends: str = ""):
        def decorator(fn):
            self.profiles.register(name, (fn, extends), allow_override=False)
            return fn
        return decorator

    # --- test isolation (review §9) ---
    @contextmanager
    def isolated(self):
        """Snapshot and restore all namespaces. Use in tests to avoid global leakage."""
        snapshots = {
            attr: copy.deepcopy(getattr(self, attr)._components)
            for attr in ("transforms", "tokenizers", "extractors",
                         "vocabularies", "profiles")
        }
        # metrics is METRIC_REGISTRY; snapshot separately
        metric_snapshot = copy.deepcopy(self.metrics.metric_metadata)
        try:
            yield
        finally:
            for attr, snap in snapshots.items():
                getattr(self, attr)._components = snap
            self.metrics.metric_metadata = metric_snapshot


REGISTRY = Registry()
```

**Namespace contracts** (review §3 — each namespace stores and returns a specific type):

| Namespace     | Stored object    | `get()` returns  | Call convention                     |
|---------------|------------------|-------------------|-------------------------------------|
| `transforms`   | `function`       | the function       | `fn(text: str, **params) -> str`    |
| `tokenizers`  | `factory fn`     | the factory        | `fn(**params) -> re.Pattern`        |
| `extractors`  | `instance`       | the instance       | `instance(dataset) -> Iterable[str]`|
| `vocabularies` | `Vocabulary` instance | the instance | already frozen; `.find_matches(text)`|
| `metrics`      | `Metric` class + metadata | class      | instantiated with `src` + params     |
| `profiles`     | `(fn, extends)`  | the function       | `fn() -> BewerConfig`               |

**Metric registry integration** (review §10): Set `REGISTRY.metrics = METRIC_REGISTRY`
in `metrics/base.py` after import. Add `get`, `list`, and `__contains__` to
`MetricRegistry`. Keep the existing `register_metric` signature with named pipeline
args + `**kwargs` param defaults. `registry.py` must not import `bewer.metrics` at
module level (import cycle); the assignment happens in `metrics/base.py`.

**Override policy** (review §9): Built-in components are registered without
`allow_override`. Attempting to re-register a built-in name raises `ValueError`.
Users must choose a different name. This prevents silently changing every profile.

### 2. Component Validation [revised: protocols are static-only, real validation at registration and resolve]

`runtime_checkable` protocols only check that `__call__` exists — they ignore
signatures (review §4). Keep them as static-typing aids only.

```python
# Static typing only — not used for runtime validation.
StandardizerFn = Callable[[str], str]
NormalizerFn = Callable[[str], str]
TokenizerFn = Callable[..., re.Pattern]
ExtractorFn = Callable[["Dataset"], Iterable[str]]
```

**Real validation** happens in two places:

**At registration** (review §4):
- Transforms: `inspect.signature(fn)` must have exactly one positional parameter
  (the text). Reject if the signature has more or fewer positional params.
- Tokenizers: all parameters must be keyword-only (or have defaults). The factory
  is never called with positional args.

**At resolve** (review §4 — preserves existing checks from `resolve.py:49-62`):
- Transforms: `inspect.signature(fn).bind(None, **params)` — reject if the first
  positional arg name appears in `params`, if required params are missing, or if
  unknown params are provided.
- Tokenizers: `inspect.signature(fn).bind(**params)` — same checks.

**Add tests for the existing resolve.py param checks before refactoring**, so they
cannot quietly disappear (review §4).

### 3. ~~VocabularyDefinition~~ Dropped [revised — review §2]

`VocabularyDefinition` is removed. The existing `Vocabulary` class is already
dataset-independent — its docstring says so, and the per-dataset state lives in the
`WeakKeyDictionary` cache.

**Registration** registers `Vocabulary` instances, frozen on registration:

```python
from importlib import resources

REGISTRY.vocabularies.register(
    "medical",
    Vocabulary("medical").add_file(resources.files("bewer.data") / "medical.txt"),
)
```

Registration freezes the vocabulary (calling `_freeze()`), the same way
`Dataset.add_vocabulary` already does. The registry key must match `vocab.name`, or
the key is derived from it.

**Auto-resolution** (review §11): When a metric references `vocab="medical"`, the
dataset checks `_vocabularies` first, then falls back to
`REGISTRY.vocabularies.get("medical")`.

**Keep `add_vocabulary`** (review §11 — do not deprecate). It is the right API for
dataset-specific lists such as the regression runner's earnings21 oracle list
(`runner.py:146-148`). Those lists do not belong in a process-wide registry.

**Consolidate vocabulary validation** (review §11): The check
`if self.vocab not in self.metric.dataset._vocabularies` is duplicated in nine
`validate()` overrides: `ktr`, `ktp`, `ktf`, `kter`, `ktcer`, `rktr`, `ktfpr`,
`_kt_stats`, `_rkt_stats`. Consolidate into a shared `KeyTermParams` base class
first. This is useful regardless of this plan.

**Report highlighting** (review §11): `Example.vocabs` lists only attached
vocabularies and drives key-term highlighting in the HTML report. With lazy
auto-resolution, whether a registered vocabulary is highlighted depends on whether
a metric over it was computed first. **Decision: resolve a profile's declared
vocabularies eagerly at `Dataset` init**, so reports are deterministic. Vocabularies
referenced by metrics but not declared in the profile are resolved lazily (and may
not appear in report highlighting — documented behavior).

**File timing** (review §11): `add_file` reads immediately and fails fast. Keep
this behavior. Vocabularies shipped with the package use `importlib.resources`.

### 4. BewerConfig Dataclass [revised: variant-level merge, no profile/overrides fields, Callable type, PipelineStep coercion, frozen, serialization as core feature]

```python
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable


@dataclass(frozen=True)
class PipelineStep:
    """A single step in a preprocessing pipeline.

    'component' is a registered name (str) or a direct callable.
    Direct callables make the config non-serializable.
    """
    component: str | Callable[..., Any]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BewerConfig:
    standardizers: dict[str, tuple[PipelineStep, ...]] = field(default_factory=dict)
    tokenizers: dict[str, PipelineStep] = field(default_factory=dict)
    normalizers: dict[str, tuple[PipelineStep, ...]] = field(default_factory=dict)
    vocabularies: dict[str, str] = field(default_factory=dict)  # name -> vocab name

    def replace(self, **kwargs) -> "BewerConfig":
        """Return a copy with the given fields replaced (review §7 mutability)."""
        return replace(self, **kwargs)
```

**Changes from the original plan (review §7):**

- **`profile` and `overrides` dropped** from the dataclass. A config is a value,
  not a recipe. The profile is chosen where the `Dataset` is created.
- **`metrics` field dropped.** It was carried into `ResolvedConfig` and never used.
  Metric presets are a separate design (see open question 3).
- **`component: str | Callable[..., Any]`** with `from __future__ import
  annotations` to avoid `TypeError` at class creation. Use `Callable[..., Any]`
  not `callable` (which is a builtin, not a type).
- **`frozen=True`** with `replace()`-style helpers. Child profiles use `replace()`
  or build a fresh object — they never mutate the parent's return value.
- **No `overrides` bag.** Overrides are just fields on the dataclass.

**Merge semantics** (review §7 — variant-level, explicitly defined):

A user-provided config is merged onto a base config at **variant granularity**:
a user variant replaces the base variant of the same name entirely. Variants the
user does not mention are inherited from the base. This differs from today's
OmegaConf merge (which is recursive within a variant — it can update a step's params
but cannot remove or reorder steps). The new rule is better: it is predictable and
lets users fully override a variant.

```python
def merge_configs(base: BewerConfig, user: BewerConfig) -> BewerConfig:
    """Merge user config onto base at variant granularity."""
    return BewerConfig(
        standardizers={**base.standardizers, **user.standardizers},
        tokenizers={**base.tokenizers, **user.tokenizers},
        normalizers={**base.normalizers, **user.normalizers},
        vocabularies={**base.vocabularies, **user.vocabularies},
    )
```

**This merge is an improvement over today's behavior**: currently a `config=` file
*replaces* `base.yml` entirely, so a custom config without `key_term` breaks KTR.
The new merge inherits unmentioned variants from the base.

**Config resolution** replaces `resolve.py`:

```python
def resolve_config(config: BewerConfig) -> Pipelines:
    """Resolve a BewerConfig to the existing Pipelines namedtuple (review §6)."""
    standardizers = {
        name: _resolve_transform_pipeline(name, steps, kind="standardizer")
        for name, steps in config.standardizers.items()
    }
    tokenizers = {
        name: _resolve_tokenizer(name, step)
        for name, step in config.tokenizers.items()
    }
    normalizers = {
        name: _resolve_transform_pipeline(name, steps, kind="normalizer")
        for name, steps in config.normalizers.items()
    }
    return Pipelines(
        standardizers=standardizers,
        tokenizers=tokenizers,
        normalizers=normalizers,
    )


def _resolve_transform_pipeline(name, steps, kind):
    """Resolve a list of PipelineSteps to a Normalizer (review §4 validation)."""
    pipeline = []
    for step in steps:
        fn = _resolve_component(step.component, "transforms")
        # Validate params against signature (preserves existing resolve.py checks)
        sig = inspect.signature(fn)
        # Check: first positional arg not in params
        # Check: required params supplied
        # Check: no unknown params
        _validate_params(fn, step.params, kind)
        pipeline.append((fn, step.params))
    return Normalizer(pipeline, name)


def _resolve_tokenizer(name, step):
    fn = _resolve_component(step.component, "tokenizers")
    _validate_params(fn, step.params, "tokenizer")
    pattern = fn(**step.params)
    return Tokenizer(pattern, name)


def _resolve_component(component, namespace):
    if isinstance(component, str):
        return getattr(REGISTRY, namespace).get(component)
    elif callable(component):
        # Direct callable — use as-is, do not auto-register (review §7).
        # Config containing direct callables is non-serializable.
        return component
    else:
        raise TypeError(f"Expected str or callable, got {type(component)}")
```

**Component name validation** happens here — typos produce immediate errors with
"did you mean?" hints (via `ComponentNotFoundError`).

### 5. Language/Domain Profiles [revised: delta profiles, split language from domain, profiles namespace in Registry]

Profiles replace `configs/languages/*.yml` with Python functions that return
**deltas** — partial `BewerConfig` objects that only specify the variants they
change. The registry builds the full chain from `extends` and merges with the
variant-level rule from §4.

```python
# src/bewer/profiles/__init__.py

@REGISTRY.register_profile("base")
def base_config() -> BewerConfig:
    return BewerConfig(
        standardizers={
            "default": (
                PipelineStep("nfc"),
                PipelineStep("normalize_apostrophe_variants"),
                PipelineStep("normalize_hyphen_variants"),
                PipelineStep("normalize_slash_variants"),
            ),
        },
        tokenizers={
            "default": PipelineStep("strip_punctuation_keep_symbols_pattern",
                                    params={"split_on_escaped": "-/"}),
            "with_punctuation": PipelineStep("keep_symbols_and_punctuation_pattern",
                                    params={"punct_chars": '.,!?:;"-/()“”«»„¡¿',
                                            "keep_newlines": True}),
            "key_term": PipelineStep("strip_punctuation_keep_symbols_pattern",
                                    params={"split_on_escaped": "-/'"}),
            "orthographically_complex_term": PipelineStep(
                "strip_punctuation_keep_symbols_pattern",
                params={"split_on_escaped": "/'"},
            ),
        },
        normalizers={
            "default": (
                PipelineStep("lowercase"),
                PipelineStep("transliterate_latin_letters"),
                PipelineStep("transliterate_symbols"),
            ),
            "cased": (
                PipelineStep("transliterate_latin_letters"),
                PipelineStep("transliterate_symbols"),
            ),
        },
    )


@REGISTRY.register_profile("da", extends="base")
def danish_delta() -> BewerConfig:
    """Only specifies what changes from base (review §8)."""
    return BewerConfig(
        normalizers={
            "default": (
                PipelineStep("lowercase"),
                PipelineStep("transliterate_latin_letters", params={"preserve": "æøå"}),
                PipelineStep("transliterate_symbols"),
            ),
        },
    )


@REGISTRY.register_profile("fr", extends="base")
def french_delta() -> BewerConfig:
    return BewerConfig(
        tokenizers={
            "default": PipelineStep("strip_punctuation_keep_symbols_pattern",
                                    params={"split_on_escaped": "-/'"}),
        },
        normalizers={
            "default": (
                PipelineStep("lowercase"),
                PipelineStep("transliterate_latin_letters",
                             params={"preserve": "àâäçéèêëîïôöùûüÿœæ"}),
                PipelineStep("transliterate_symbols"),
            ),
        },
    )
```

**Profile resolution** (review §8 — registry builds the chain from `extends`):

```python
def resolve_profile(name: str) -> BewerConfig:
    fn, extends = REGISTRY.profiles.get(name)
    delta = fn()
    if extends:
        base = resolve_profile(extends)
        return merge_configs(base, delta)
    return delta
```

**Split language from domain** (review §8 — open question 1 resolved):

Language profiles decide preprocessing. Domain profiles decide vocabularies and
metric presets. They are separate axes, combined where the `Dataset` is created:

```python
dataset = Dataset(
    profile="da",                          # language: preprocessing
    vocabularies=["medical"],               # domain: vocabularies
)
```

No `medical_da` profile needed. No multiple inheritance.

**`with_punctuation` punct_chars** (review §13): The base profile must use the
exact Unicode characters from `base.yml:14`, including curly quotes U+201C/U+201D
(`""`), not ASCII `"`. The parity test (§13 below) catches this.

### 6. Dataset Changes [revised: config+language together, PathLike for files, Pipelines return, clone/config attrs, eager vocab resolution]

```python
from os import PathLike

class Dataset:
    def __init__(
        self,
        config: BewerConfig | str | PathLike | None = None,
        language: str | None = None,
        *,
        profile: str | None = None,
        vocabularies: list[str] | None = None,
    ):
        # Profile selection (review §6: keep language= as alias)
        prof = profile or language

        if isinstance(config, BewerConfig):
            self._config = config
            if prof:
                # Merge language profile onto user config (review §6)
                self._config = merge_configs(resolve_profile(prof), config)
        elif isinstance(config, (str, PathLike)):
            if config in REGISTRY.profiles:
                self._config = resolve_profile(config)
            elif prof:
                self._config = merge_configs(resolve_profile(prof), from_yaml(config))
            else:
                self._config = from_yaml(config)
        elif config is None and prof is not None:
            self._config = resolve_profile(prof)
        else:
            self._config = resolve_profile("base")

        # Eager vocabulary resolution (review §11: deterministic reports)
        self._config = self._config.replace(
            vocabularies={**(self._config.vocabularies or {}),
                          **{v: v for v in (vocabularies or [])}}
        )

        # Resolve to Pipelines namedtuple (review §6: not ResolvedConfig)
        self._pipelines = resolve_config(self._config)
        self.config_path = None  # set if loaded from file
        self._init_blank_state()

        # Eagerly attach declared vocabularies (review §11)
        for name in self._config.vocabularies:
            self._resolve_vocabulary(name)

    def _resolve_vocabulary(self, name: str) -> Vocabulary:
        """Resolve a vocabulary by name — attached or registered."""
        if name in self._vocabularies:
            return self._vocabularies[name]
        if name in REGISTRY.vocabularies:
            vocab = REGISTRY.vocabularies.get(name)
            self._register_derived_vocabulary(vocab)
            return vocab
        raise ValueError(
            f"Vocabulary '{name}' not found. "
            f"Attached: {sorted(self._vocabularies)}, "
            f"Registered: {REGISTRY.vocabularies.list()}"
        )
```

**Fixes from review §6:**

- **`config` + `language` together**: language profile is merged onto the user's
  config, preserving today's behavior.
- **`str` ambiguity**: `PathLike` is a separate type for file paths. A string is
  checked against `REGISTRY.profiles` first; if not found, it is treated as a file
  path. A file named `da` in the working directory collides — documented limitation.
  Recommend `os.PathLike` for file paths.
- **`clone()`**: `self.config.copy()` is an OmegaConf method. Replace with
  `replace(self._config)` (dataclass) or `copy.deepcopy(self._config)`.
- **`config` / `config_path` public attributes**: `self.config` returns the
  `BewerConfig`; `self.config_path` is set to the file path if loaded from a file,
  `None` otherwise. Tests that assert these (`test_dataset.py:20`) are updated.
- **Return type**: `resolve_config()` returns the existing `Pipelines` namedtuple,
  not a new `ResolvedConfig`.
- **`language=`** kept as alias for `profile=` (regression runner `runner.py:142`
  and ~10 tests use it).

### 7. Pipeline Variant Validation [revised — review §1: in MetricCollection.get(), not resolve_config()]

```python
# In MetricCollection.get() — after resolve_params(), before freeze():
def metric_factory(**kwargs):
    resolved = METRIC_REGISTRY.resolve_params(name, **kwargs)

    # Validate variant names against the dataset's pipelines (review §1)
    _validate_pipeline_variants(resolved, self._src.pipelines)

    # ... existing cache check ...

    metric_instance = METRIC_REGISTRY.create_metric(name, src=self._src, **kwargs)
    self._src.freeze()  # only after validation passes
    # ...


def _validate_pipeline_variants(resolved, pipelines):
    """Check that resolved standardizer/tokenizer/normalizer names exist."""
    for stage_name, pipeline_dict in [
        ("standardizer", pipelines.standardizers),
        ("tokenizer", pipelines.tokenizers),
        ("normalizer", pipelines.normalizers),
    ]:
        variant = resolved.get(stage_name, DEFAULT)
        if variant not in pipeline_dict:
            raise ValueError(
                f"{stage_name.capitalize()} '{variant}' not found. "
                f"Available: {sorted(pipeline_dict)}"
            )
```

This also catches the case where a custom config does not define a variant that a
built-in metric depends on through its registration defaults (`key_term`,
`with_punctuation`, `orthographically_complex_term`, `cased`).

**Bug fix** (review §1): A failed metric request currently freezes the dataset
anyway (verified). The validation must happen *before* `self._src.freeze()` so a
failed request does not freeze the dataset. The existing "failed metric request
does not freeze" test invariant is preserved.

### 8. What Does NOT Change

- **`Pipelines` namedtuple** — same structure, just built differently.
- **ContextVar mechanism** (`context.py`) — unchanged.
- **`pipeline_cached_property`** (`caching.py`) — unchanged.
- **`Text.standardized`, `Text.tokens`, `Token.normalized`** — unchanged.
- **`Metric` base class, `MetricParams`, `metric_value`, `dependency`** — unchanged.
- **`MetricCollection`, `ExampleMetricCollection`** — unchanged in shape (the variant
  validation in §7 is added inside `get()`, not a new class).
- **`Vocabulary`** — unchanged. No `VocabularyDefinition` wrapper (review §2).
- **`Vocabulary` resolution cache** (`WeakKeyDictionary`) — unchanged.
- **`KeyTermTrie`** — unchanged.
- **Reporting system** — affected only by serialized config embedding (§9).
- **`set_pipeline` context manager** — unchanged.
- **`add_vocabulary`** — kept, not deprecated (review §11).
- **`flags.py`** — kept (review §14: at most move constants into `context.py`).

### 9. YAML: Single Reader with Dotted-Path Fallback [revised — review §12: core feature, no separate shim, embed in reports]

**Serialization is a core feature**, not optional (review §12). For an evaluation
framework, recording exactly which normalization produced a WER number is the main
payoff of naming components.

```python
def to_yaml(config: BewerConfig) -> str:
    """Serialize a BewerConfig to YAML using registry names.

    Direct callables (not registered) raise SerializationError.
    """
    ...


def from_yaml(yaml_str: str | PathLike) -> BewerConfig:
    """Parse YAML into a BewerConfig.

    Accepts registry names (e.g. 'lowercase') and, as a permanent fallback,
    dotted paths (e.g. 'bewer.preprocessing.normalization.lowercase').
    Any name containing '.' is treated as a dotted path and imported (review §12).
    """
    ...
```

**Single YAML format, not two** (review §12). The `from_yaml` reader accepts both
registry names and dotted paths permanently. Dotted paths let users plug in custom
functions without registering them. No separate backward-compat shim.

**Embed serialized config in reports and regression baselines** (review §12):
- HTML reports include the serialized config in a `<details>` block.
- Regression baselines record the config alongside metric values.

**Drop `OmegaConf` and `hydra-core`** from `pyproject.toml` (review §12: nothing
imports hydra-core). Use `pyyaml` (already a dependency) for YAML I/O.

---

## Implementation Steps

Each phase is a PR that ships on its own, with the regression suite green.

### Phase 0: Validation & Consolidation (no registry, no config changes) [revised — review §1, §4, §11]

1. **Validate variant names in `MetricCollection.get()`** before freezing
   (review §1). ~10 lines. Closes TODO.md:52-62. Ship as its own PR.

2. **Consolidate the nine duplicated vocabulary `validate()` overrides** into a
   shared `KeyTermParams` base class (review §11). Useful regardless of this plan.

3. **Add tests for the existing `resolve.py` param checks** (review §4): first
   positional arg not in params, required params supplied, unknown params rejected.
   These tests currently don't exist (`grep` finds nothing under `tests/`).

4. **Fix the frozen-on-failure bug** in `MetricCollection.get()`: a failed metric
   request currently freezes the dataset (verified). The validation in step 1 must
   happen before `self._src.freeze()`.

### Phase 1: Registry Core (additive, no breaking changes) [revised — review §3, §4, §5, §9, §10]

5. **Create `src/bewer/registry.py`** with `Registry`, `_ComponentRegistry`,
   `ComponentNotFoundError`, `REGISTRY` singleton. Include `isolated()` context
   manager for test isolation (review §9).

6. **Register all existing preprocessing functions** as transforms (single
   namespace, review §5):
   ```python
   @REGISTRY.transforms.register(length_preserving=True)
   def nfc(text: str) -> str:
       return unicodedata.normalize("NFC", text)

   @REGISTRY.transforms.register
   def lowercase(text: str) -> str:
       return text.lower()

   @REGISTRY.transforms.register(token_only=True)
   def strip_punctuation(text: str, *, ...) -> str:
       ...
   ```
   Move `_set_attrs` metadata (`token_only`, `length_preserving`) onto registration
   as keyword args to `register()`. Default name to `fn.__name__`.

7. **Register tokenizer factories** in `tokenization.py`:
   ```python
   @REGISTRY.tokenizers.register
   def strip_punctuation_keep_symbols_pattern(split_on_escaped=None) -> re.Pattern:
       ...
   ```

8. **Register extractor instances** (review §3 — instances, not classes):
   ```python
   REGISTRY.extractors.register(
       "orthographically_complex",
       OrthographicallyComplexTermExtractor(),
   )
   ```

9. **Set `REGISTRY.metrics = METRIC_REGISTRY`** in `metrics/base.py` after import
   (review §10). Add `get`, `list`, `__contains__` to `MetricRegistry`. Keep the
   existing `register_metric` signature with named pipeline args + `**kwargs` param
   defaults. `registry.py` must not import `bewer.metrics` at module level.

10. **Add signature-shape validation at registration** (review §4): transforms
    must have exactly one positional parameter; tokenizer factories must have
    keyword-only or defaulted parameters.

11. **Add `REGISTRY.isolated()` pytest fixture** in `conftest.py` (review §9).
    All registry-touching tests use it to prevent global-state leakage.

12. **Update the YAML loader** (`from_yaml`) to accept registry names alongside
    dotted paths (review §12). Any name containing `.` is imported; otherwise looked
    up in the registry. This is the single permanent YAML reader.

13. **Add tests** for the registry: registration, lookup, duplicate detection,
    listing, `ComponentNotFoundError` with "did you mean?" hints, `isolated()`.

### Phase 2: BewerConfig & Profiles (breaking change: Dataset init) [revised — review §6, §7, §8, §11, §13]

14. **Create `src/bewer/config.py`** with `BewerConfig` (frozen dataclass),
    `PipelineStep`, `merge_configs()`, `resolve_config()`, `resolve_profile()`.

15. **Create `src/bewer/profiles/__init__.py`** with `base`, `en` (= empty delta),
    `da`, `de`, `fr` delta profiles, registered via `@REGISTRY.register_profile()`.

16. **Add `bewer.profiles` to `bewer/__init__.py` imports** (review §14) so profiles
    register at import time.

17. **Move base pipeline definitions** from `configs/base.yml` to the `base` profile.
    **Copy byte-for-byte** — the parity test (step 21) catches transcription errors
    like the curly-quote issue (review §13).

18. **Update `Dataset.__init__`** to accept `BewerConfig | str | PathLike | None`,
    `language=` (alias for `profile=`), and `vocabularies=` list. Fix `clone()`,
    `config`, and `config_path` attributes (review §6).

19. **Add `Dataset._resolve_vocabulary(name)`** for auto-resolution from registry.
    Eagerly resolve declared vocabularies at init (review §11).

20. **Update metric param validation** to use `Dataset._resolve_vocabulary(name)`
    instead of direct `dataset._vocabularies` lookup. The consolidated
    `KeyTermParams` base from Phase 0 makes this one change.

21. **Add a parity test** (review §13): for every language, the Python profile must
    resolve to the same pipelines as the YAML — the same function objects and params
    per step, and equal tokenizer `pattern.pattern` strings. This test exists only
    while YAML and Python profiles coexist.

22. **Add `da` to the regression manifest** (review §13 — it is the only profile
    with no baseline). Run the full regression suite and commit baselines.

23. **Add tests** for config resolution, profile loading, merge semantics, pipeline
    component validation, vocabulary auto-resolution.

### Phase 3: Cleanup & Serialization [revised — review §12, §14]

24. **Remove YAML config files** (`configs/base.yml`, `configs/languages/*.yml`).
    The `from_yaml` reader still accepts dotted paths for backward compatibility.

25. **Remove `OmegaConf`** from `pyproject.toml`. Remove `hydra-core` (review §12:
    nothing imports it). Remove the parity test (step 21) — it has no YAML to compare
    against.

26. **Remove `configs/resolve.py`**. The `_resolve_function()` and
    `importlib.import_module` string resolution are replaced by registry lookups
    in `config.py`. (The `from_yaml` dotted-path fallback has its own minimal
    import logic.)

27. **Add `to_yaml()`** in `config.py`. Embed serialized config in HTML reports
    (`<details>` block) and regression baselines (review §12).

28. **Update `TODO.md`** — mark "Validate pipeline names" and "Config-based metric
    registration" as resolved (or leave the latter open if presets are not yet
    implemented — see open question 3).

29. **Update `README.md`** and `CLAUDE.md`** — replace YAML references with Python
    config examples.

30. **Full test pass** — ensure all existing tests pass and new tests cover the
    registry, config, and profile systems.

### Phase 4: Vocabulary Registry & Metric Presets (separate design) [revised — review §2, §8, open question 3]

This phase is intentionally separate. It depends on Phases 1–3 being stable.

31. **Register built-in vocabularies** as `Vocabulary` instances (review §2):
    ```python
    REGISTRY.vocabularies.register(
        "orthographically_complex_terms",
        Vocabulary("orthographically_complex_terms")
        .add_extractor(REGISTRY.extractors.get("orthographically_complex")),
    )
    ```

32. **Update `orthographically_complex_term.py`** to use
    `REGISTRY.vocabularies.get("orthographically_complex_terms")` instead of
    `_ensure_orthographically_complex_term_vocabulary()`.

33. **Design metric presets** (open question 3, review §8): a per-dataset alias table
    in `MetricCollection` that maps a name to `(base_metric, params)`. Build the cache
    key from the resolved base name and params, so `my_wer()` and `wer(normalizer=...)`
    share one instance and one freeze path. A preset must not shadow a registered
    metric name. This deserves its own short design and PR.

34. **Add tests** for vocabulary registration, auto-resolution, and presets.

---

## File Changes Summary [revised]

| File | Phase | Action |
|------|-------|--------|
| `src/bewer/registry.py` | 1 | **New** — unified registry, `ComponentNotFoundError`, `isolated()` |
| `src/bewer/config.py` | 2 | **New** — `BewerConfig`, `PipelineStep`, `merge_configs()`, `resolve_config()`, `resolve_profile()`, `to_yaml()`, `from_yaml()` |
| `src/bewer/profiles/__init__.py` | 2 | **New** — language delta profiles |
| `src/bewer/core/dataset.py` | 0, 2, 4 | **Modify** — variant validation, config loading, vocabulary auto-resolution |
| `src/bewer/core/vocabulary.py` | — | Unchanged (no `VocabularyDefinition`) |
| `src/bewer/preprocessing/normalization.py` | 1 | **Modify** — add `@REGISTRY.transforms.register` decorators, remove `_set_attrs` |
| `src/bewer/preprocessing/tokenization.py` | 1 | **Modify** — add `@REGISTRY.tokenizers.register` decorators |
| `src/bewer/metrics/base.py` | 0, 1 | **Modify** — variant validation in `get()`, `REGISTRY.metrics = METRIC_REGISTRY`, add `get`/`list`/`__contains__` |
| `src/bewer/metrics/orthographically_complex_term.py` | 4 | **Modify** — use registry vocab resolution |
| `src/bewer/metrics/_kt_stats.py`, `_rkt_stats.py`, `ktr.py`, etc. | 0 | **Modify** — consolidate `validate()` into `KeyTermParams` base |
| `src/bewer/extractors/orthographically_complex_term.py` | 1 | **Modify** — register extractor instance |
| `src/bewer/extractors/regex.py` | — | Unchanged (base class) |
| `src/bewer/__init__.py` | 1, 2 | **Modify** — export `REGISTRY`, `BewerConfig`; import `bewer.profiles` |
| `src/bewer/configs/base.yml` | 3 | **Remove** |
| `src/bewer/configs/languages/*.yml` | 3 | **Remove** |
| `src/bewer/configs/resolve.py` | 3 | **Remove** |
| `src/bewer/flags.py` | — | **Keep** (at most move constants into `context.py`) |
| `pyproject.toml` | 3 | **Modify** — remove `omegaconf`, `hydra-core` |
| `tests/` | 0–4 | **Modify** — add registry tests, config tests, parity test, migrate fixtures |
| `tests/conftest.py` | 1 | **Modify** — add `REGISTRY.isolated()` fixture |
| `regression/manifest.yml` | 2 | **Modify** — add `da` baseline |
| `TODO.md`, `README.md`, `CLAUDE.md` | 3 | **Modify** |

---

## Risk Assessment [revised — adds test leakage, report ordering, metric drift]

**Low risk:**
- Phase 0 is ~10 lines + a consolidation. No registry, no config changes.
- Phase 1 is additive — existing code keeps working. The YAML loader accepts
  registry names alongside dotted paths.
- ContextVar/caching mechanism is completely orthogonal — zero changes.
- Metric registration API is backward compatible (`REGISTRY.metrics = METRIC_REGISTRY`,
  existing decorator signature preserved).

**Medium risk:**
- `Dataset.__init__` signature change (Phase 2) — mitigated by accepting
  `BewerConfig | str | PathLike | None` and keeping `language=`.
- Vocabulary auto-resolution changes the metric creation flow — need to ensure
  `_register_derived_vocabulary` path still works for late registration.
- Profile delta semantics are a new concept — the parity test and the regression
  suite are the safety net.

**What could go wrong (revised):**
- **Global-state test leakage** (review §9): a test that registers a component
  leaks into every later test. Mitigated by `REGISTRY.isolated()` fixture.
- **Report output depends on computation order** (review §11): with lazy
  auto-resolution, a registered vocabulary may not appear in report highlighting
  unless a metric over it was computed first. Mitigated by eager vocabulary
  resolution at `Dataset` init for declared vocabularies.
- **Metric drift from transcription errors** (review §13): the original plan had
  a curly-quote → ASCII-quote error in `with_punctuation`'s `punct_chars` that
  would silently change PER. Mitigated by the parity test and the regression suite.
- **Import cycle**: `registry.py` must not import `bewer.metrics` at module level.
  The assignment `REGISTRY.metrics = METRIC_REGISTRY` happens in `metrics/base.py`.
- **`str` ambiguity** (review §6): a file named `da` in the working directory
  collides with the profile. Documented limitation; recommend `os.PathLike` for
  file paths.

---

## Open Questions [revised — all answered]

1. **~~Profile composition~~** — Resolved (review §8): no multiple inheritance. Split
   language (preprocessing) from domain (vocabularies, presets) and combine them
   where the `Dataset` is created: `Dataset(profile="da", vocabularies=["medical"])`.

2. **~~`VocabularyDefinition` mutability~~** — Resolved (review §2): moot.
   `VocabularyDefinition` is dropped. Register `Vocabulary` instances, which
   already freeze when attached; freeze them on registration as well.

3. **Metric presets** — Resolved (review §8): yes, define them as presets — a
   per-dataset alias table in `MetricCollection` that maps a name to
   `(base_metric, params)`. Build the cache key from the resolved base name and
   params so `my_wer()` and `wer(normalizer=...)` share one instance. A preset must
   not shadow a registered metric name. This deserves its own short design and PR
   (Phase 4), not a field that is passed through unused.

4. **~~YAML backward compat~~** — Resolved (review §12): no separate shim. Write a
   single `from_yaml` that accepts registry names and, as a permanent fallback,
   dotted paths (any name containing `.` is imported). Drop `OmegaConf` and
   `hydra-core`.

---

## What the Plan Keeps from the Original

- Phase 1 is additive and keeps the metric API stable.
- The ContextVar and `pipeline_cached_property` machinery is untouched.
- Merging user config onto a base fixes a real gap (custom configs can drop
  variants that built-in metrics require).
- A Python config is easier to type-check and compose than dotted-path YAML.
- Typo errors that list available names are the right user experience — they
  belong in both the metric factory (Phase 0) and config resolution (Phase 2).
