# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

BeWER (Beyond Word Error Rate) is an evaluation and analysis framework for automatic speech recognition (ASR) in Python. It provides a Python-based configuration system for evaluation pipelines with metrics computation, preprocessing, and web-based reporting.

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
- Supports loading data from CSV, pandas DataFrames (with planned HuggingFace support)
- Configuration system based on BewerConfig (regular class, frozen on registration or dataset construction) with Python configs

**Example** (`src/bewer/core/example.py`)
- Represents a single reference-hypothesis pair
- Contains Text objects (ref/hyp) and optional keywords
- Each Example has its own MetricCollection for per-example metrics

**Text** (`src/bewer/core/text.py`)
- Immutable text representation with preprocessing pipeline
- Stores original and standardized text and caches tokenization results
- Normalization is applied per-token via the active normalizer pipeline
- Lazy evaluation of preprocessing stages

**Preprocessing Pipeline** (`src/bewer/preprocessing/`)
- Three-stage pipeline: standardization → tokenization → token-level normalization
- Configured via Python configs (`src/bewer/configs/`)
- Each stage is a series of function applications (Transforms) wrapped in pipeline objects (Standardizer, Tokenizer, Normalizer)
- Standardizers: Unicode normalization (NFC)
- Tokenizers: Whitespace-based with customizable symbol handling
- Normalizers: Lowercase, transliteration, symbol removal/translation (applied to tokens)

### Metrics System

**MetricCollection** (`src/bewer/metrics/base.py`)
- Provides attribute-based access to metrics: `dataset.metrics.wer.value`
- Lazy computation: metrics computed on first access
- Automatic caching to avoid recomputation
- Supports both dataset-level and example-level metrics

**Built-in Metrics** (`src/bewer/metrics/`)
- `general/`: WER (`wer.py`), CER (`cer.py`), PER (`per.py`); B-WER (`b_wer.py`) and U-WER (`u_wer.py`) — WER on words in/out of a `Vocabulary` (Le et al., 2021 contextual-biasing protocol), both delegating to the shared hidden `_bu_wer_outcomes` (`_bu_wer_outcomes.py`); `multi_token` param selects occurrence/flag/split semantics for multi-token terms
- `alignment/`: Levenshtein distance (`levenshtein.py`), error alignment (`error_align.py`, uses external error-align package)
- `key_term/`: Key term recall/precision/F-score/CER/etc. (`ktr.py`, `ktp.py`, `ktf.py`, `ktfpr.py`, `ktcer.py`, `rktr.py`, `kter.py`); shared stats (`_kt_stats.py`, `_rkt_stats.py`); orthographically-complex term metrics (`orthographically_complex_term.py`)
- `hallucination/`: Insertion rate (`insertion_rate.py`) — counts insertions in contiguous runs of length >= `min_run_length` (default 1), normalized by reference token count. `min_run_length>=2` counts only burst runs (hallucination signal). `alignment` parameter selects backend (`'levenshtein'` or `'error_align'`). Example insertion rate (`example_insertion_rate.py`) — per-example variant.
- `summary/`: Summary metric (`summary.py`)
- Orthographically-complex term metrics register KTR/KTP/KTF/KTFPR with a different tokenizer (`orthographically_complex_term`, no hyphen split), `cased` normalizer (no lowercasing), and an `orthographically_complex_terms` vocabulary auto-extracted from references (acronyms, alphanumerics, hyphen compounds; e.g. `MRI`, `HbA1c`, `CT-scan`), defined in the base config.

### Key Term Package (`src/bewer/key_term/`)

Groups all key-term machinery:

- **`key_term.py`**: `KeyTerm` (a `Text` subclass for key terms) and `KeyTermMatch` (a match result — token span, source text, matched key term).
- **`trie.py`**: `KeyTermTrie` — the Aho-Corasick matching engine. `build_matcher` produces one; `find_matches(text)` returns `KeyTermMatch` objects; `key_terms` exposes the input terms.
- **`vocabulary.py`**: `Vocabulary` — a named collection of key terms (static lists, files, extractor functions). `ExtractorFn` protocol, `VocabularyFrozenError`, `VocabularyExtractorError`.
- **`regex_extractor.py`**: `RegexExtractor` — generic base that full-matches each reference token against a compiled pattern. Subclass it (override `default_pattern`) or instantiate with a `pattern` to define a new regex-based term family. Its matching primitive `match_token_regex` returns unit token slices.
- **`orthographically_complex_term_extractor.py`**: `OrthographicallyComplexTermExtractor` — a `RegexExtractor` with `ORTHOGRAPHICALLY_COMPLEX_TERM_DEFAULT_PATTERN`, backing the orthographically-complex term metrics. Expects the `orthographically_complex_term` tokenizer (no hyphen split) so `CT-scan` is a single token.
- Exposed at the package top level as `bewer.key_term`.

### Alignment System (`src/bewer/alignment/`)

- Text alignment for error analysis (insertions, deletions, substitutions)
- Operation types defined in `op_type.py`
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

Configuration is managed through `BewerConfig` (a regular class, frozen on registration or dataset construction) and Python configs:

- Config class: `src/bewer/configs/config.py` — `BewerConfig`, `get_config()`
- Configs: `src/bewer/configs/{base,en,da,de,fr}.py` — built-in config presets, each defined as a module-level constant and registered via `CONFIG_REGISTRY.register(name, config)`
- Registry: `src/bewer/configs/config.py` — `ConfigRegistry` / `CONFIG_REGISTRY` for config presets; metrics use `METRIC_REGISTRY` in `bewer.metrics.base`
- Extensible: users can provide custom `BewerConfig` instances or register custom configs
- Language configs extend `BASE_CONFIG` via the `extends` keyword argument
- Vocabularies are attached to configs via `config.add_vocabulary(vocab)` (requires `.clone()` first if the config is frozen)
- `ConfigFrozenError` raised on mutation of a frozen config; `DatasetFrozenError` and `VocabularyFrozenError` for the other freezable objects

Example config structure:
```python
from bewer import BewerConfig, Transform
from bewer.core.vocabulary import Vocabulary
from bewer.preprocessing.normalization import Normalizer, Standardizer, lowercase, nfc
from bewer.preprocessing.tokenization import Tokenizer, strip_punctuation_keep_symbols_pattern

config = BewerConfig(
    standardizers=(
        Standardizer("default", Transform(nfc)),
    ),
    tokenizers=(
        Tokenizer("default", strip_punctuation_keep_symbols_pattern(split_on_escaped="-/")),
    ),
    normalizers=(
        Normalizer("default", Transform(lowercase)),
    ),
    vocabularies=(
        Vocabulary("medical", terms=["diabetes", "asthma"]),
    ),
)
```

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

## Dependencies

**Core Dependencies**:
- pandas: Data handling
- regex, rapidfuzz: Text processing and matching
- error-align: External alignment library (Corti package)
- jinja2: HTML template rendering
- rich: CLI output formatting

**Build System**:
- Uses both Poetry (development) and Hatch (packaging)
- Version managed by hatch-vcs from git tags
- Supports Python 3.10-3.14

## Important Notes

- The preprocessing pipeline is immutable and lazy - Text objects cache results
- Metrics are computed lazily and cached - avoid manual cache invalidation
- Key terms need not appear in the references; terms that do not simply contribute no
  reference-side counts (the old KeyTermNotFoundWarning no longer exists)
- The project uses semantic versioning via git tags (hatch-vcs)
- Pre-commit hooks include poetry-lock which auto-updates on pyproject.toml changes

## Versioning & Releases

The project follows SemVer with the pre-1.0 convention:

- **Breaking change**: bump **minor** (`0.2.3 → 0.3.0`).
- **New feature or bug fix**: bump **patch** (`0.2.3 → 0.2.4`).
- Pre-release suffixes (`rcN`, `bN`) stage an upcoming release only.
- `1.0.0` marks API stability; after that, standard SemVer applies (breaking → major).
- Versions are derived from git tags via `hatch-vcs`. Do not hardcode a version.

PR label → version bump mapping:

- `major` label = breaking change → minor bump (pre-1.0).
- `minor` label = new feature → patch bump.
- `patch` label = fix → patch bump.

This mapping lives in `.github/release_drafter_main_branch.yml` and must be reverted to one-to-one (`major→major`, etc.) when releasing `1.0.0`.
