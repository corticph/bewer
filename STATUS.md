# Unified Registry & Python Config — Status Report

**Branch:** `feature/unified-registry`  
**PR:** [#100](https://github.com/corticph/bewer/pull/100)  
**Base:** `main` (025b1c2)  
**Tests:** 1175 passing  
**Date:** 2026-10-09  

## Goal

Replace the YAML/OmegaConf configuration system with a pure-Python `BewerConfig` dataclass and a unified `REGISTRY` singleton for all pluggable components (metrics, configs). Preprocessing functions and vocabularies are now passed directly in config, not registered.

## What Changed (in order)

### Phase 0 — Validation & consolidation (51e1707)
- Added `_validate_pipeline_variants` in `MetricCollection.get()` — typos in standardizer/tokenizer/normalizer variant names now raise `ValueError` immediately, before `freeze()`.
- Consolidated 9 duplicate `validate()` overrides across key-term metrics into a shared `KeyTermParams` base class in `metrics/base.py`.
- Added param tests for `resolve.py`.

### Phase 1 — Registry core (c01ccc7, 668a8e6)
- Created `src/bewer/registry.py` with `ComponentRegistry` (name→object dict with `register`/`get`/`list`/`__contains__`) and `Registry` singleton with typed namespaces.
- Registered all existing transforms, tokenizers, and the OCT extractor via decorators.
- Added `isolated()` context manager for test isolation.
- Added YAML registry-name resolution (bare names resolve via REGISTRY, dotted paths via importlib).
- Added transform signature validation at registration time.
- Exported `REGISTRY` from package top level.

### Phase 2 — BewerConfig & profiles (d57c9ac)
- Created `src/bewer/config.py` with `BewerConfig` (frozen dataclass), `PipelineStep`, `merge_configs()`, `resolve_config()`, `resolve_profile()`, `to_yaml()`, `from_yaml()`.
- Created `src/bewer/profiles/__init__.py` with `base`, `en`, `da`, `de`, `fr` profiles registered via `@REGISTRY.register_profile`.
- Updated `Dataset.__init__` to accept `config: BewerConfig | str | None`.

### Phase 3 — Remove YAML/OmegaConf (6c38f2d)
- Removed all YAML config files (`configs/languages/*.yml`).
- Removed `configs/` package, `OmegaConf`, `hydra-core` from dependencies.
- Moved `Pipelines` namedtuple from `configs/resolve.py` to `config.py`.
- Updated `CLAUDE.md` and `README.md`.

### Phase 4 — Vocabulary registry & OCT (28a2676, 4fc445e, 011765e)
- Initially added `REGISTRY.vocabularies` and `REGISTRY.extractors` namespaces.
- Later **dropped both** — vocabularies now live in `BewerConfig.vocabularies` as `dict[str, Vocabulary]`.
- OCT vocabulary added to base config (eagerly attached at `Dataset` init).
- OCT metrics simplified: removed all subclasses and custom param classes; now just three `register_metric` calls with `vocab`, `tokenizer`, `normalizer` defaults.

### Post-Phase 4 simplifications (user-driven)

1. **Dropped preprocessing function registration** (e816666) — transforms and tokenizers are no longer registered in REGISTRY. `defaults.py` imports them directly and passes callables.

2. **Replaced `PipelineStep` with `Transform`** (9167271) — `**params` instead of `params=` dict, `__call__` method for direct invocation. Tokenizers are bare `re.Pattern` objects from direct factory calls (no wrapper).

3. **Removed YAML round-trip** (fed5038) — Dropped `to_yaml`, `from_yaml`, `SerializationError`, all YAML helper functions, and the `pyyaml` dependency. Config is pure Python.

4. **Removed importlib string resolution** (d8c2cf0) — `_resolve_component` and dotted-path string support are gone. Config always holds callables.

5. **Moved param validation into `Transform.__init__`** (0967478) — `_validate_params` runs at construction time, not at resolution time.

6. **Renamed `Transform.component` → `Transform.fn`** (327fb8d) — shorter, matches Normalizer convention.

7. **Slimmed `ComponentRegistry`** (1ac573d) — Removed `validate_fn`, `**attrs`, `allow_override`, `__iter__`, `__len__`, and the bare-decorator form. Only direct-call `register` and `get`/`list`/`__contains__` remain.

8. **Removed `ComponentNotFoundError`** (5087914) — `get()` now raises plain `KeyError` with available names listed. No `difflib` dependency.

9. **Dropped `REGISTRY.extractors` and `REGISTRY.vocabularies`** (28a2676) — Vocabularies are in `BewerConfig.vocabularies`. The OCT extractor is used directly when building the vocabulary in `defaults.py`.

10. **Renamed "profiles" to "configs"** (a93164d) — `profiles/__init__.py` → `defaults.py`, `resolve_profile()` → `get_config()`, `REGISTRY.profiles` → `REGISTRY.configs`, `register_profile` → `REGISTRY.configs.register`. The term "profile" is gone.

11. **Slimmed `Dataset.__init__`** (e41fb61) — Removed `language=` and `profile=` params. Only `config: BewerConfig | str | None` remains (str = registered config name).

12. **Converted `Pipelines` to frozen dataclass** (a7ab666) — Was a namedtuple; now a `@dataclass(frozen=True)`.

13. **Fixed Unicode escapes** — `defaults.py` uses actual characters (`""«»„¡¿`) instead of `\uXXXX` escapes.

### Copilot review comments on PR #100
All 7 comments addressed (replies posted on GitHub, threads resolved):
- **#1 BewerConfig mutability** — deferred (known limitation, no code mutates config post-construction)
- **#2 `_validate_params` treats `**kwargs` as required** — fixed (skip `VAR_POSITIONAL`/`VAR_KEYWORD`)
- **#3 `to_yaml` drops duplicate components** — fixed (reject with error), later removed entirely with YAML
- **#4 Callable serialization checks wrong namespace** — fixed, later removed entirely with YAML
- **#5 Merge order reversed** — stale (code removed)
- **#6 `BewerConfig.vocabularies` alias not implemented** — fixed (changed to `dict[str, Vocabulary]`)
- **#7 `MetricRegistry` missing `get()`** — fixed

## Current State

### Registry (`src/bewer/registry.py`, ~107 lines)
The `REGISTRY` singleton has only two namespaces:
- `REGISTRY.configs` — registered config factories (`base`, `en`, `da`, `de`, `fr`) with `extends` chain support
- `REGISTRY.metrics` — `MetricRegistry` (lazily property to avoid circular imports)

`ComponentRegistry` supports: `register(name, obj, extends=...)` (direct call only), `get(name)`, `list()`, `__contains__`, and `isolated()` on the `Registry` class.

### Config (`src/bewer/config.py`, ~150 lines)
- `BewerConfig` — frozen dataclass with `standardizers`, `tokenizers`, `normalizers` (dicts of variant-name → tuple of `Transform`s or `re.Pattern`), and `vocabularies` (`dict[str, Vocabulary]`).
- `Transform` — callable wrapper with `**params`, validates params at construction, `__call__` for direct use.
- `resolve_config(BewerConfig)` → `Pipelines` (frozen dataclass of `Normalizer`/`Tokenizer` instances).
- `get_config(name)` — resolves a registered config name through its `extends` chain to a `BewerConfig`.
- `merge_configs(base, delta)` — merges two `BewerConfig` instances.

### Defaults (`src/bewer/defaults.py`)
- Five config presets: `base`, `en` (empty delta), `da`, `de`, `fr` — registered via `@REGISTRY.configs.register(name, extends=parent)`.
- Imports transform functions and tokenizer factories directly from `preprocessing/`.
- OCT vocabulary declared in base config via `Vocabulary.add_extractor(OrthographicallyComplexTermExtractor())`.

### Dataset (`src/bewer/core/dataset.py`)
- `Dataset(config: BewerConfig | str | None, ...)` — str is a registered config name.
- Vocabularies from config are eagerly attached at init.
- `_resolve_vocabulary(name)` checks attached vocabularies, then raises `KeyError`.
- `add_vocabulary(vocab)` still exists for attaching a `Vocabulary` instance to a dataset at runtime.

### OCT metrics (`src/bewer/metrics/orthographically_complex_term.py`, ~40 lines)
- Three `register_metric` calls: `KTR`, `KTP`, `KTF` with `vocab="orthographically_complex_terms"`, `tokenizer="orthographically_complex_term"`, `normalizer="cased"`.
- No subclasses, no custom param classes, no metadata.

## What's NOT in scope / Open items
- `add_vocabulary` on `Dataset` is retained but could be deprecated now that vocabularies live in config.
- `ComponentRegistry` is public but only used for `configs` (could be simplified further if desired).
- No YAML serialization — configs are shared as Python code.
- `CLAUDE.md` and `README.md` were updated in Phase 3 but may not fully reflect all post-Phase 4 simplifications (Transform, dropped REGISTRY namespaces, etc.).
- Design docs (`docs/design/`) were removed from the branch.
