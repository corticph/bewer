# Review: Unified Registry & Python Configuration plan

Review of [unified-registry.md](unified-registry.md), checked against the code on
`feature/surface-view-toggle` (95347f5). Wherever this review says "verified", the behaviour was
reproduced by running it against the current code.

## Verdict

The direction is sound. One naming scheme for components, a typed Python config, and earlier
validation are all worth having. The plan is not ready to implement as written, for five reasons:

1. **The headline fix is in the wrong place.** Pipeline-name typos cannot be caught by
   `resolve_config()`. The real fix is about 10 lines in `MetricCollection.get()` and needs no registry.
2. **`VocabularyDefinition` is based on a misreading of `Vocabulary`.** The existing class already
   is a dataset-independent definition. Phase 3 can mostly be dropped.
3. **Several code sketches would not work as written.** These are the extractor and vocabulary
   registration, the protocols, the `PipelineStep` type hint, and the `with_punctuation` profile.
4. **Phase 2 bundles three independent changes** (config, profiles and vocabulary
   auto-resolution) and depends on a class that is only created in Phase 3.
5. **Nothing checks that metrics stay identical.** The repo already has the tool for this
   (`regression/`), and the plan already contains a transcription error that would silently change PER.

The findings below are ranked by impact. After them come answers to the open questions and a
revised phasing.

---

## 1. Pipeline-name validation cannot happen in `resolve_config()`

The plan mixes up two separate namespaces:

| Namespace | Examples | Defined by | Chosen when |
|---|---|---|---|
| **Component names** | `lowercase`, `nfc`, `strip_punctuation_keep_symbols_pattern` | registry | in the config |
| **Pipeline variant names** | `default`, `cased`, `key_term`, `with_punctuation` | config | at metric request: `metrics.wer(normalizer=...)` and metric registration defaults |

The TODO item and the error message in §6 (`Normalizer 'defualt' not found. Available: ['cased',
'default']`) concern **variant** names. A typo like `normalizer="defualt"` is passed to
`dataset.metrics.wer(...)`, long after the config has been resolved, so `resolve_config()` never
sees it. `resolve_config()` can only validate **component** names.

Current behaviour (verified):

```python
ds.metrics.wer(normalizer="defualt")   # succeeds AND freezes the dataset
m.value                                # ValueError: 'defualt' not found in normalizers.
```

A failed request also freezes the dataset, which breaks the "failed metric request does not
freeze" invariant that the tests protect for unknown params.

**Recommendation.** In `MetricCollection.get()`, after `resolve_params()` and before
`self._src.freeze()` ([base.py:538-566](../../src/bewer/metrics/base.py#L538-L566)), check that
each resolved standardizer, tokenizer and normalizer exists in `self._src.pipelines`, and raise
with the list of available names. This closes the TODO item without a registry, so ship it first
as its own PR. It also catches the case where a custom config does not define a variant that a
built-in metric depends on through its registration defaults (`key_term`, `with_punctuation`,
`orthographically_complex_term`, `cased`).

Rewrite §6 ("Pipeline name validation moves to `resolve_config()`") and the "Key improvement"
note in §4 so they name both namespaces and say where each one is validated.

A smaller correction: the "Current State" table says preprocessing has "no existence check until
import". The import already runs inside `Dataset.__init__` (via `resolve_pipelines`), so a missing
function already fails at construction. What the registry improves is the error message
(`ModuleNotFoundError`/`AttributeError` becomes "X not found, available: …"), not when the error
happens.

## 2. `VocabularyDefinition` duplicates what `Vocabulary` already is

§3 calls the current `Vocabulary` "per-dataset, resolved". It is not. Its docstring
([vocabulary.py:46-55](../../src/bewer/core/vocabulary.py#L46-L55)) says it "holds only its
*definition*; the concrete key terms are resolved lazily per dataset … a single Vocabulary can be
attached to and shared across multiple datasets". The per-dataset state lives in the
`WeakKeyDictionary` cache. The plan's own `VocabularyDefinition.resolve(dataset)` never uses
`dataset`.

The sketches also do not work (verified):

- `class MedicalVocabulary(VocabularyDefinition)` with class attributes `name`/`terms`. The
  inherited dataclass `__init__` still requires `name`, so `MedicalVocabulary()` raises
  `TypeError: missing 1 required positional argument: 'name'`.
- The decorator registers the **class**, so `REGISTRY.vocabularies.get("medical").resolve(self)`
  calls an unbound method and raises `TypeError: resolve() missing 1 required positional argument:
  'dataset'`.
- §1 and §3 show two different shapes for a registered vocabulary: a plain class with
  `terms = [...]`, and a `VocabularyDefinition` subclass.
- The builder example calls `.add_terms()` and `.add_extractor()` on a frozen dataclass that
  defines neither method.

**Recommendation.** Drop `VocabularyDefinition`. Register `Vocabulary` **instances** and freeze
them on registration, the same way `add_vocabulary` already does:

```python
REGISTRY.vocabularies.register(Vocabulary("medical").add_file(resources.files("bewer.data") / "medical.txt"))
```

This removes a class, open question 2, and most of Phase 3. Check that the registry key matches
`vocab.name`, or derive the key from it.

## 3. Registries need a defined "what does `get()` return" contract per namespace

The namespaces store different kinds of objects, and the plan never says which:

| Namespace | Stored object | How it's used |
|---|---|---|
| metrics | class | instantiated with `src` + params |
| tokenizers | factory | called once with params → pattern → `Tokenizer` |
| normalizers/standardizers | function | called per text with params |
| extractors | **class in the plan**, but used as an instance | called with `dataset` |

The extractor case is a real bug. `@REGISTRY.register_extractor` decorates the class
`OrthographicallyComplexTermExtractor`, and `VocabularyDefinition.resolve` hands
`REGISTRY.extractors.get(name)` to `add_extractor`. Calling the class with a dataset builds an
extractor whose `pattern` is the dataset. Verified: this fails with
`TypeError: 'OrthographicallyComplexTermExtractor' object is not iterable`.

**Recommendation.** Register extractor instances, for example
`REGISTRY.extractors.register("orthographically_complex", OrthographicallyComplexTermExtractor())`.
Then state the stored type and call convention for each namespace in §1.

## 4. The protocols do not validate anything, and the real checks get dropped

`runtime_checkable` protocols only check that `__call__` exists. They ignore signatures. Verified:
`isinstance(lambda a, b, c: 1, TokenizerFn)` and `isinstance(str.lower, ExtractorFn)` are both
`True`. `StandardizerFn` and `NormalizerFn` are identical. So the §2 claim that "registration
validates against these protocols (with clear error messages)" does not hold.

At the same time, the plan's `_resolve_step` returns `(callable, params)` without any checks. That
drops the validation that exists today
([resolve.py:49-62](../../src/bewer/configs/resolve.py#L49-L62)): the first positional argument
must not be in params, required params must be supplied, and unknown params are rejected. None of
these checks has a test today (`grep` finds nothing under `tests/`).

**Recommendation.** Keep the protocols as static-typing aids only. Validate in two places:

- **At registration:** check the signature shape. Transforms take exactly one positional `text`
  parameter. Tokenizer factories take keyword-only parameters.
- **At resolve:** call `inspect.signature(fn).bind(None, **params)` for transforms and
  `.bind(**params)` for factories.

Add tests for the existing checks *before* refactoring, so they cannot quietly disappear.

## 5. Standardizers and normalizers should share one namespace

Both stages are `str -> str`, and both resolve to the same `Normalizer` class
([resolve.py:66](../../src/bewer/configs/resolve.py#L66)). `nfc` lives in `normalization.py` but
is used as a standardizer. Two separate namespaces would force double registration, or block reuse
that YAML allows today.

**Recommendation.** Use a single `transforms` namespace. Move the existing `_set_attrs` metadata
(`token_only`, `length_preserving`, at
[normalization.py:33-43](../../src/bewer/preprocessing/normalization.py#L33-L43)) onto
registration: `register_transform("lowercase", length_preserving=True)`. The metadata then gives
real validation the plan currently lacks:

- reject `token_only` functions in a standardizer pipeline;
- let `ErrorAlign` check up front whether the chosen normalizer is length-preserving, instead of
  warning per token at runtime ([error_align.py:71-94](../../src/bewer/metrics/error_align.py#L71-L94)).

Also default the registered name to `fn.__name__` so a bare `@REGISTRY.register_transform` works.
That avoids writing every name twice.

## 6. `Dataset.__init__` silently changes behaviour

- **`config` + `language` together.** Today the language overlay is merged onto the user's
  config ([dataset.py:51-53](../../src/bewer/core/dataset.py#L51-L53)). Verified: a custom config
  plus `language="da"` gets `preserve: "æøå"`. In the plan's branch, `language` is ignored whenever
  `config` is given.
- **`str` is ambiguous.** A string could be a profile name or a file path, and a file named `da`
  in the working directory would collide with the profile. Accept `os.PathLike` for files, or use
  separate keyword arguments.
- **`clone()`** calls `self.config.copy()` ([dataset.py:106](../../src/bewer/core/dataset.py#L106)),
  which is an OmegaConf method that a dataclass does not have. The public `config`/`config_path`
  attributes (asserted in `tests/test_core/test_dataset.py:20`) are not addressed either.
- **Return-type mismatch.** `resolve_config()` returns `ResolvedConfig(standardizers=…,
  tokenizers=…, …)`, but `Dataset` reads `self._resolved.pipelines`. Have it return the existing
  `Pipelines` namedtuple.
- **Keep `language=`** as an alias for `profile=`. The regression runner (`runner.py:142`) and
  about 10 tests use it.

## 7. `BewerConfig` semantics are underspecified

- **Merge semantics.** Every field defaults to `{}`, so `merge_configs` cannot tell "not set" from
  "set to empty". Define the merge at variant granularity: a user variant replaces the base variant
  of the same name entirely, and variants the user does not mention are inherited. This differs
  from today's OmegaConf merge, which is recursive within a variant (it can update a step's params
  but cannot remove or reorder steps). The new rule is better but must be written down. The merge
  is itself an unadvertised improvement: today a `config=` file *replaces* `base.yml`, so a custom
  config without `key_term` breaks KTR.
- **Drop `profile` and `overrides` from the dataclass.** Including them makes the config both a
  value and a recipe (a profile returns a config that could name another profile), and
  `overrides: dict[str, Any]` is an untyped bag that defeats the purpose of a typed config. Choose
  the profile where the `Dataset` is created.
- **`metrics: dict[str, dict]` is carried into `ResolvedConfig` and never used**, yet step 23
  marks "Config-based metric registration" as resolved. Either design presets (see open
  question 3) or leave the TODO open.
- **`component: str | callable`** raises `TypeError` when the class is created (verified), unless
  the module uses `from __future__ import annotations`. Use `Callable[..., Any]`. Coerce `str` and
  callables to `PipelineStep` in `__post_init__`, so downstream code handles one type instead of
  three.
- **Direct callables.** The comment "auto-register under a generated name" would modify global
  state as a side effect of resolving a config. Use the callable as-is and mark the config as not
  serializable.
- **Mutability.** Child profiles mutate the object their parent returns. That is safe only while
  every call builds a fresh object, and breaks as soon as someone caches a profile. Use
  `frozen=True` with `replace()`-style helpers, or deep-copy in `REGISTRY.profiles.get()`.

## 8. Profiles: `extends` is decorative and will drift

Inheritance works by calling the parent function, and `extends=` is "purely for documentation".
Nothing stops `extends="da"` from sitting on a body that calls `base_config()`.

**Recommendation.** Have a profile function return a *delta*, and let the registry build the chain
from `extends` and merge with the rule from §7. This matches today's overlay files one to one
(`da.yml` is six lines) and keeps language profiles short. Alternatively, drop `extends`.

`medical_da` shows the combinatorial problem coming: language × domain. Language decides
preprocessing; domain decides vocabularies and metric presets. Keep them as separate axes and
combine them where the `Dataset` is created, for example
`Dataset(profile="da", vocabularies=["medical"])`. Open question 1 then goes away.

Also, `REGISTRY.profiles` and `register_profile` are used in §4–§6 but missing from the
`Registry` class in §1.

## 9. A global mutable registry needs test isolation

Any test that registers a component leaks into every later test, through duplicate-name errors or,
worse, silent overrides. Add `REGISTRY.isolated()` (a context manager that snapshots and restores
the registry) and a pytest fixture in Phase 1. Also decide the `allow_override` policy for built-ins.
Overriding `lowercase` would silently change every profile, so consider forbidding it or at least
warning.

`_ComponentRegistry.get` raises `KeyError`, which wraps its message in quotes, while §6 shows a
`ValueError`. Use a dedicated `ComponentNotFoundError(LookupError)`, and add a
`difflib.get_close_matches` "did you mean …?" hint, which costs little and suits the typo case.

## 10. The metric registry migration is described circularly

Step 4 says `MetricRegistry.register_metric()` delegates to `REGISTRY.metrics.register()`, and
also that `METRIC_REGISTRY` becomes a thin layer pointing at `REGISTRY.metrics`. It never says
which object owns `metric_metadata`, `resolve_params` and `create_metric`, which
`MetricCollection.get`, `list_metrics`, `_format_registered_params` and `list_registered_metrics`
all read.

**Recommendation.** Set `REGISTRY.metrics = METRIC_REGISTRY` and add `get`, `list` and
`__contains__` to `MetricRegistry`. That needs no wrapper and changes no behaviour. Also keep the
existing decorator signature. The plan's `register_metric(name, **pipeline_defaults)` drops
*param* defaults: today `**kwargs` are param defaults and the pipeline defaults are named
arguments ([base.py:672-681](../../src/bewer/metrics/base.py#L672-L681)).

`registry.py` must not import from `bewer.metrics` at module level, because `metrics/base.py` will
import `REGISTRY`, and that would create an import cycle.

## 11. Vocabulary auto-resolution touches more than step 12 says

- **The vocabulary check is duplicated nine times.** Step 12 says to update
  "`MetricParams.validate()`", but `vocab in dataset._vocabularies` is checked in nine separate
  `validate()` overrides: `ktr`, `ktp`, `ktf`, `kter`, `ktcer`, `rktr`, `ktfpr`, `_kt_stats` and
  `_rkt_stats`. Consolidate them into one shared `KeyTermParams` base first. That is useful
  regardless of this plan.
- **Reports depend on computation order.** `Text.get_key_term_matches` silently returns `[]` for
  an unknown vocab ([text.py:146](../../src/bewer/core/text.py#L146)). `Example.vocabs`
  ([example.py:70](../../src/bewer/core/example.py#L70)) lists only attached vocabularies, and it
  drives key-term highlighting in the HTML report
  ([alignment.py:201](../../src/bewer/reporting/html/alignment.py#L201)). With lazy
  auto-resolution, whether a registered vocabulary is highlighted depends on whether a metric over
  it was computed first. Choose one: resolve a profile's declared vocabularies eagerly at
  `Dataset` init, or keep it lazy and document the behaviour.
- **Do not deprecate `add_vocabulary`.** Step 18 contradicts §3 and §6, which keep it for one-off
  vocabularies. It is also the right API for dataset-specific lists such as the regression
  runner's earnings21 oracle list ([runner.py:146-148](../../regression/runner.py#L146-L148)).
  Those lists do not belong in a process-wide registry.
- **Phase order is broken.** Steps 10–12 in Phase 2 use `VocabularyDefinition`, which Phase 3
  creates.
- **File timing changes.** `files: tuple[str, ...]` re-reads paths relative to the working
  directory on every resolve. Today `add_file` reads immediately and fails fast. Vocabularies
  shipped with the package need `importlib.resources`.

## 12. YAML: two dialects, and the compatibility shim contradicts Phase 4

- **The shim needs what Phase 4 removes.** The shim has to read dotted paths
  (`bewer.preprocessing.normalization.lowercase`), so it needs `_resolve_function` and
  `import_module`, which step 22 removes. The File Changes table also removes `resolve.py` in
  Phase 4, while step 20 keeps it "temporarily".
- **Two YAML formats.** Phase 5 then adds a second YAML dialect based on registry names, so the
  project would have two formats for the same thing.
- **Recommendation (open question 4).** The project is pre-1.0 and documents that breaking changes
  may occur. Write a single `from_yaml` that accepts registry names and, as a permanent fallback,
  dotted paths (any name containing `.` is imported). That one reader covers backward
  compatibility and also lets users plug in custom functions without registering them. Drop the
  separate shim.
- **Make serialization a core feature, not "optional, low priority".** For an evaluation
  framework, recording exactly which normalization produced a WER number is the main payoff of
  naming components at all. Embed the serialized config in HTML reports and regression baselines.
  Without serialization, a registry for preprocessing functions is hard to justify over passing
  callables directly.
- **Drop `hydra-core`** from `pyproject.toml` together with OmegaConf. Nothing imports it.

## 13. No gate checks that metrics stay identical

For the preprocessing migration, the only acceptable outcome is identical metric values. The plan's
§5 already contains a transcription error. `base.yml:14` has curly quotes (U+201C, U+201D) in
`with_punctuation`'s `punct_chars`, and the plan's Python version has ASCII `"` instead (verified
byte by byte). Copying the plan as written would silently change PER tokenization.

**Recommendation.** Add two gates:

1. **A parity test, while YAML and Python profiles coexist.** For every language, the Python
   profile must resolve to the same pipelines as the YAML: the same function objects and params per
   step, and equal tokenizer `pattern.pattern` strings.
2. **The existing regression suite.** `regression/runner.py`, with committed baselines for en
   (earnings21), de and fr, must pass unchanged before Phase 4 deletes the YAML. Add `da` to the
   manifest first, since it is the only profile with no baseline.

## 14. Smaller items

- **`flags.py` removal (step 26) is churn with no payoff.** `DEFAULT` is not a registry concept.
  At most, move the constants into `preprocessing/context.py`.
- **Discovery is still import-time registration.** It works because `bewer/__init__.py` imports
  every submodule; the new `bewer.profiles` must be added there too. Entry points for third-party
  plugins can come later. Mark them out of scope.
- **"What does NOT change" lists the metric registration API as backward compatible**, but the
  sketched `register_metric` signature is not (see §10).
- **The Risk Assessment omits** global-state test leakage (§9), report output that depends on
  computation order (§11), and metric drift from transcription errors (§13).

---

## Answers to the open questions

1. **Profile composition.** No multiple inheritance. Split language (preprocessing) from domain
   (vocabularies, metric presets) and combine them where the `Dataset` is created (§8).
2. **`VocabularyDefinition` mutability.** Moot. Register `Vocabulary` instances, which already
   freeze when attached; freeze them on registration as well (§2).
3. **Metric presets.** Yes, define them as presets: a per-dataset alias table in
   `MetricCollection` that maps a name to `(base_metric, params)`. Build the cache key from the
   resolved base name and params, so `my_wer()` and `wer(normalizer=...)` share one instance and
   one freeze path. A preset must not shadow a registered metric name. This deserves its own short
   design and PR, not a field that is passed through unused.
4. **How long to keep the YAML shim.** Do not add one. Write a single YAML reader that accepts
   dotted paths permanently (§12).

## Suggested revised phasing

Each phase should be a PR that can ship on its own, with the regression suite green.

| Phase | Scope | User-visible change |
|---|---|---|
| **0** | Validate variant names in `MetricCollection.get()` before freezing (§1). Consolidate the nine vocabulary `validate()`s (§11). Add tests for the existing `resolve.py` param checks (§4). | Typo errors at request time. Closes the TODO item. |
| **1** | Registry with a single `transforms` namespace plus `tokenizers`, extractor instances, and `metrics = METRIC_REGISTRY`. Signature checks at registration, `_set_attrs` metadata, `isolated()` fixture. The YAML loader accepts registry names alongside dotted paths. | None (additive). |
| **2** | `BewerConfig` and delta profiles with the variant-level merge. `Dataset(config: BewerConfig \| PathLike, profile/language: str)`. Parity test against YAML. `clone()` and `config` attributes fixed. | Python config available. YAML still works. |
| **3** | Delete the packaged YAML, OmegaConf and hydra-core. `to_yaml`/`from_yaml` become the one YAML path. Serialized config recorded in reports. | YAML format uses registry names; dotted paths still accepted. |
| **4** (separate design) | Vocabulary registry (instances) with an explicit eager or lazy rule; metric presets. | Named vocabularies and presets. |

## What the plan gets right

- Phase 1 is additive and keeps the metric API stable.
- It correctly leaves the ContextVar and `pipeline_cached_property` machinery alone. Nothing here
  needs to touch it.
- Merging user config onto a base fixes a real gap: today, custom configs can drop variants that
  built-in metrics require.
- A Python config is easier to type-check and compose than dotted-path YAML.
- Typo errors that list the available names are the right user experience. They just belong in
  the metric factory (§1) as well as in config resolution.
