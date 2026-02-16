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
- Configuration system based on OmegaConf with YAML config files

**Example** (`src/bewer/core/example.py`)
- Represents a single reference-hypothesis pair
- Contains Text objects (ref/hyp) and optional keywords
- Each Example has its own MetricCollection for per-example metrics

**Text** (`src/bewer/core/text.py`)
- Immutable text representation with preprocessing pipeline
- Stores original, standardized, normalized, and tokenized versions
- Lazy evaluation of preprocessing stages

**Preprocessing Pipeline** (`src/bewer/preprocessing/`)
- Three-stage pipeline: standardization → tokenization → normalization
- Configured via YAML (`src/bewer/configs/base.yml`)
- Each stage is a series of function applications
- Standardizers: Unicode normalization (NFC)
- Tokenizers: Whitespace-based with customizable symbol handling
- Normalizers: Lowercase, transliteration, symbol removal/translation

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
- Legacy Corti metrics: `corti_legacy_metrics.py`

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

Configuration is managed through YAML files with OmegaConf/Hydra:

- Default config: `src/bewer/configs/base.yml`
- Defines preprocessing pipelines (standardizers, tokenizers, normalizers)
- Extensible: users can provide custom configs
- Pipeline resolution happens in `configs/resolve.py`

Example config structure:
```yaml
standardizers:
  default:
    bewer.preprocessing.normalization.nfc:

tokenizers:
  default:
    bewer.preprocessing.tokenization.whitespace_strip_symbols_and_custom:
      split_on: "-/"

normalizers:
  default:
    bewer.preprocessing.normalization.lowercase:
    bewer.preprocessing.normalization.transliterate_latin_letters:
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
- pyyaml, omegaconf, hydra-core: Configuration management
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
- Keywords must exist in reference text or a warning is issued
- The project uses semantic versioning via git tags (hatch-vcs)
- Pre-commit hooks include poetry-lock which auto-updates on pyproject.toml changes
