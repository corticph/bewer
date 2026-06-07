# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BeWER (Beyond Word Error Rate) is an evaluation and analysis framework for automatic speech recognition (ASR) in Python. It provides a YAML-based configuration system for evaluation pipelines with metrics computation, preprocessing, and web-based reporting.

**Status**: Early development, not production-ready. Breaking changes may occur between alpha versions.

## Development Commands

### Setup
```bash
make install  # Install dependencies with Poetry and set up pre-commit hooks
```

### Testing
```bash
make test                    # Run full test suite with coverage reports
poetry run pytest tests/     # Run all tests
poetry run pytest tests/test_core/test_dataset.py  # Run specific test file
poetry run pytest tests/test_metrics/ -k "test_wer"  # Run tests matching pattern
```

### Linting and Formatting
```bash
make pre-commit              # Run all pre-commit hooks
make pre-commit-pipeline     # Run pre-commit hooks scoped to the preprocessing pipeline
poetry run ruff check .      # Run ruff linter
poetry run ruff format .     # Format code with ruff
```

### Cleanup
```bash
make clean  # Remove __pycache__, coverage reports, and test artifacts
```

### Building and Publishing
```bash
poetry build                 # Build distribution packages
poetry run twine check dist/*  # Validate built packages
```

## Architecture

### Core Components

**Dataset** (`src/bewer/core/dataset.py`)
- Main entry point for the framework
- Manages collections of Examples and provides lazy metric computation
- Loads data via `load_csv`, `load_jsonl`, and `load_pandas`; `load_dataset` (HuggingFace) is a stub that raises `NotImplementedError`
- Registers vocabularies: `add_vocabulary`, `add_vocabulary_from_list`, `add_vocabulary_from_file`, `add_vocabulary_from_function`
- Configuration system based on OmegaConf with YAML config files

**Example** (`src/bewer/core/example.py`)
- Represents a single reference-hypothesis pair
- Contains Text objects (ref/hyp) and optional key terms
- Each Example has its own MetricCollection for per-example metrics

**Text / TokenizedText** (`src/bewer/core/text.py`)
- `TokenizedText` is the base class providing standardization, tokenization, and pipeline caching; `Text` (a reference or hypothesis) and `KeyTerm` are its two subclasses
- Immutable: stores original and standardized text and caches tokenization/normalization results
- Normalization is applied per-token via the active normalizer pipeline; preprocessing stages are evaluated lazily
- Also defines `TextType` (ref/hyp/key-term enum) and `TokenList`

**Token** (`src/bewer/core/token.py`)
- A single token: raw string, char offsets/slice, position index, and lazily-normalized form

**KeyTerm** (`src/bewer/core/key_term.py`)
- A term to locate within reference tokens; a sibling of `Text` under `TokenizedText`, canonicalized (deduped by raw string) per owning `Vocabulary`
- `Match` records a located span; matching uses an Aho-Corasick trie (`pyahocorasick`)

**Vocabulary** (`src/bewer/core/vocabulary.py`)
- A named source of key terms that locates them in text. Terms come from explicit lists/files (`from_list`/`from_file`), lazy extraction functions (`from_function`), and per-example annotations — combined as a union and matched via one Aho-Corasick trie
- `VocabularyExtractor` is a `(dataset) -> terms` callable: returning `Iterable[str]` yields *global* terms; returning `Mapping[int, Iterable[str]]` yields per-example *local* terms (see `bewer.extractors`)
- `only_local_matches` is a matching-time policy controlling whether a term matches everywhere or only in the examples that regard it

**Pipeline caching** (`src/bewer/core/caching.py`)
- `pipeline_cached_property`: a context-sensitive descriptor that caches preprocessing-stage results keyed by the active pipeline

**Preprocessing Pipeline** (`src/bewer/preprocessing/`)
- Three-stage pipeline: standardization → tokenization → token-level normalization, each a series of YAML-configured function applications (`src/bewer/configs/base.yml`)
- The active pipeline stage is tracked through context vars in `context.py`
- Standardizers & normalizers (`normalization.py`): NFC plus apostrophe/hyphen/slash variant normalization (standardize stage); lowercase, Latin transliteration, symbol transliteration/removal (normalize stage)
- Tokenizers (`tokenization.py`): regex-pattern tokenizers — punctuation-stripping, symbol-keeping, plain-whitespace, and legacy variants. Named tokenizer configs in `base.yml` include `default`, `key_term`, and `complex_term` (like `key_term` but does not split on hyphens, for strict complex-term scoring)

### Metrics System

**MetricCollection** (`src/bewer/metrics/base.py`)
- Provides attribute-based access to metrics: `dataset.metrics.wer.value`
- Lazy computation: metrics computed on first access
- Automatic caching to avoid recomputation
- Supports both dataset-level and example-level metrics

**Built-in Metrics** (`src/bewer/metrics/`)
- WER (Word Error Rate): `wer.py`
- CER (Character Error Rate): `cer.py`
- Levenshtein distance: `levenshtein.py`
- Error alignment metrics: `error_align.py` (uses external error-align package)
- Key-term metrics (KTR/KTP/KTF): recall/precision/F-score over a key-term vocabulary — `ktr.py`, `ktp.py`, `ktf.py`
- Key-term error-rate and CER variants (KTER/KTCER): `kter.py`, `ktcer.py`
- Relaxed key-term recall (RKTR): `rktr.py`
- Complex-term metrics (CTR/CTP/CTF): `complex_term.py` — subclass the key-term metrics over auto-extracted complex terms (acronyms, alphanumerics, hyphen compounds, Greek-bearing tokens); auto-register a `complex_terms` vocabulary on first use, so `dataset.metrics.ctr().value` works out of the box. They run under the `complex_term` tokenizer (no hyphen splitting) and the `cased` normalizer (no lowercasing), so a term's surface form is scored strictly — neither `CT scan` nor `ct-scan`/`mri` match a `CT-scan`/`MRI` term (unlike word-level metrics)
- Dataset summary (DatasetSummary): `summary.py`
- Confidence intervals (ConfidenceInterval): `confidence.py`
- Legacy Corti metrics: `corti_legacy_metrics.py`

### Vocabulary Extractors (`src/bewer/extractors/`)

Library of pre-defined `VocabularyExtractor` callables — `(dataset) -> terms` functions that derive key terms on demand and are registered via `Dataset.add_vocabulary_from_function` (or wrapped with `Vocabulary.from_function`).

- **`ComplexTermExtractor`** (`extractors/complex_term.py`): scans each example's *reference* for complex terms (acronyms, alphanumerics, hyphen compounds, Greek-bearing tokens), returning a per-example (local) vocabulary that backs the complex-term metrics (CTR/CTP/CTF). It reads `example.ref.tokens` under the active tokenizer and expects the `complex_term` pipeline (no hyphen split), so a compound like `CT-scan` is a single token
- Its regex matching primitive lives in the same module: `match_token_regex` (full-matches each token against a compiled pattern, returning unit slices) and `COMPLEX_TERM_DEFAULT_PATTERN`. No cross-token grouping — compounds arrive whole from the `complex_term` tokenizer
- Exposed at the package top level as `bewer.extractors`

### Alignment System (`src/bewer/alignment/`)

- Text alignment for error analysis (insertions, deletions, substitutions)
- `Alignment` (`alignment.py`): a tuple of `Op`s representing the aligned operation sequence
- `Op` (`op.py`): a single alignment operation; operation types (`OpType`) defined in `op_type.py`
- Used by metrics and reporting components

### Reporting (`src/bewer/reporting/`)

**HTML Reporting** (`reporting/html/`)
- Jinja2-based HTML report generation
- Templates in `src/bewer/templates/`
- Alignment visualization with color schemes
- Gallery views for inspecting individual examples

**Python Reporting** (`reporting/python/`)
- Programmatic access to alignment tables and analysis
- Utilities for integration with pandas and other tools

## Configuration System

Configuration is managed through YAML files with OmegaConf:

- Default config: `src/bewer/configs/base.yml`; language-specific overrides in `src/bewer/configs/languages/` (`da`, `de`, `en`, `fr`)
- Defines preprocessing pipelines (standardizers, tokenizers, normalizers)
- Extensible: users can provide custom configs
- Pipeline resolution happens in `configs/resolve.py`
- Section-key constants (`standardizers`/`tokenizers`/`normalizers`/`default`) live in `src/bewer/flags.py`

Structure: three top-level sections — `standardizers`, `tokenizers`, `normalizers` — each mapping a pipeline name (`default`, `key_term`, …) to an ordered list of `bewer.preprocessing.*` functions with optional kwargs. Rather than duplicate the pipeline definitions here (which drifts out of date), see [`src/bewer/configs/base.yml`](src/bewer/configs/base.yml) for the canonical, current set and the available named pipelines.

## Testing Conventions

- Tests mirror the source structure (`tests/test_core/`, `tests/test_metrics/`, etc.)
- Use pytest with typeguard for runtime type checking
- Coverage tracking enabled by default (minimum coverage enforced in CI)
- Fixtures defined in `tests/conftest.py`

## Code Style

- Line length: 120 characters (configured in `ruff.toml`)
- Ruff linting with E, F, and I rules enabled
- Pre-commit hooks enforce formatting, linting, and security checks
- Type hints expected (py.typed marker present)
- Method/attribute visibility follows the Python convention that a leading underscore (`_foo`) marks a
  member as internal to its **own class** — i.e. only accessed via `self`. Anything called or read across
  an object boundary (by a collaborating class) must be public, with no leading underscore.

## Dependencies

**Core Dependencies**:
- pandas: Data handling
- regex, rapidfuzz, levenshtein: Text processing, fuzzy and edit-distance matching
- pyahocorasick: Aho-Corasick trie for key-term / vocabulary matching
- unidecode: Latin transliteration in the normalizer pipeline
- pyyaml, omegaconf, hydra-core: Configuration management
- error-align: External alignment library (Corti package)
- jinja2: HTML template rendering
- rich: CLI output formatting
- typeguard: Runtime type checking
- fuzzywuzzy: Temporary, for legacy Corti metrics (to be removed)

**Build System**:
- Uses both Poetry (development) and Hatch (packaging)
- Version managed by hatch-vcs from git tags
- Supports Python 3.10-3.14

## Important Notes

- The preprocessing pipeline is immutable and lazy - Text objects cache results
- Metrics are computed lazily and cached - avoid manual cache invalidation
- Keywords must exist in reference text or a warning is logged
- The project uses semantic versioning via git tags (hatch-vcs)
- Pre-commit hooks include poetry-lock which auto-updates on pyproject.toml changes
- Always update the README and this CLAUDE.md with any architectural or workflow changes to keep documentation current
