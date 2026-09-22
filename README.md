
<img src="https://raw.githubusercontent.com/corticph/bewer/main/.github/assets/logo.svg" alt="BeWER" width="100%"/>

<p align="center">
  <img src="https://img.shields.io/badge/python-%203.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-green" alt="Python Versions">
  <img src="https://codecov.io/gh/corticph/bewer/graph/badge.svg?token=4QBH8TD4T4" alt="Coverage" style="margin-left:5px;">
  <img src="https://img.shields.io/pypi/v/bewer" alt="PyPI">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" style="margin-left:5px;">
</p>

<br>

**⚠️ Important:** This project is not production ready and is still in early development. Breaking changes may occur, and backwards compatibility between alpha versions is not guaranteed.

**Bewer is an evaluation and analysis framework for automatic speech recognition in Python.** It defines a transparent YAML-based approach for configuring evaluation pipelines and makes it easy to inspect and analyze individual examples through a web-based interface. The built-in preprocessing pipeline and metrics collection are designed to cover all conventional use cases and then some, while still being fully extensible.




__Contents__ | [Installation](#installation) | [Quickstart](#quickstart) | [Metrics](#metrics) |


<a name="installation">

## Installation

```bash
pip install bewer
```

<a name="quickstart">

## Quickstart

```python
from bewer import Dataset

# Create an evaluation dataset
dataset = Dataset(language="en")

# Load data
dataset.load_csv(
    "data.csv",
    ref_col="reference",
    hyp_col="hypothesis",
)

# List available metrics and compute
dataset.metrics.list_metrics()
print(f"WER: {dataset.metrics.wer().value:.2%}")
```

<a name="metrics">

## Metrics Catalog

| | Type | Accessor | |
|--------|------|----------|---:|
| **General purpose** | | | |
| Word Error Rate | General | `wer` | [`>`](src/bewer/metrics/wer.py) |
| Character Error Rate | General | `cer` | [`>`](src/bewer/metrics/cer.py) |
| **Key-term metrics** | | | |
| Key-Term Recall | Key-term | `ktr` | [`>`](src/bewer/metrics/ktr.py) |
| Key-Term Precision | Key-term | `ktp` | [`>`](src/bewer/metrics/ktp.py) |
| Key-Term F-Score | Key-term | `ktf` | [`>`](src/bewer/metrics/ktf.py) |
| Key-Term Error Rate | Key-term | `kter` | [`>`](src/bewer/metrics/kter.py) |
| Key-Term False-Positive Rate | Key-term | `ktfpr` | [`>`](src/bewer/metrics/ktfpr.py) |
| Key-Term Character Error Rate | Key-term | `ktcer` | [`>`](src/bewer/metrics/ktcer.py) |
| Relaxed Key-Term Recall | Key-term | `rktr` | [`>`](src/bewer/metrics/rktr.py) |
| **Alignments** | | | |
| Levenshtein Alignment | Alignment | `levenshtein` | [`>`](src/bewer/metrics/levenshtein.py) |
| Error Alignment | Alignment | `error_align` | [`>`](src/bewer/metrics/error_align.py) |

## Core Concepts

### Hierarchy

In `bewer`, evaluation is centered around the `Dataset` object, which implements a linguistic hierarchy, from a collection of reference-hypothesis pairs to individual tokens.

```python
# Create a dataset and populate it
dataset = Dataset(language="en")
dataset.add(ref="foo", hyp="bar")

# Climb down the hierarchy: Dataset -> Example -> Text -> Token
example = dataset[0]
text = example.ref
token = text.tokens[0]
```

### Preprocessing Pipeline

Text preprocessing is a central part of an evaluation run and typically require specific considerations for different languages, domains, or tasks. When you set the language of the dataset, the **preprocessing pipeline** is updated accordingly. The preprocessing pipeline is divided into three steps: standardization, tokenization, and normalization.

```python
text.raw                # String: Original text
text.standardized       # String: Standardized text
token = text.tokens[0]
token.raw               # String: Standardized token
token.normalized        # String: Normalized token
```

A brief overview of the preprocessing steps:
- **Standardization** is intended to iron out inconsistencies that may affect tokenization. This includes basic normalization of characters that separate tokens, but should also take into account abbreviations and numerical formats.
- **Tokenization** is regex-based, which allows for tracing individual tokens back to their positions in the standardized text. While typically not essential for metric computation, it makes post-hoc analysis easier.
- **Normalization** is applied at the token level and typically take care of lowercasing, removing diacritics, and other language-specific adjustments. For most metrics, normalization can be toggled on or off as needed.

The available preprocessing steps for a given language configuration can be inspected via the `pipelines` attribute of the dataset.





## Metrics

**Lazy evaluation and caching.** Metrics are computed lazily and cached, so requesting the same metric with the same parameters twice returns the cached result. Requesting a metric *freezes* the dataset: its contents can no longer change, so further `add()`/`load_*()` calls raise `DatasetFrozenError`. Use `clone()` for a fresh, modifiable copy to keep building.

```python
# When a metric is initialized, it is cached and the dataset is frozen.
wer = dataset.metrics.wer()
assert wer is dataset.metrics.wer()
assert wer is not dataset.clone().metrics.wer()
```
**Metric registry.** All `bewer` metrics are registered in the metric registry under one or more accesor names with different configurations. For instance, key-term recall is registered in a general form under the accessor name `ktr`, but also comes with pre-defined vocabulary specifications (e.g., `orthographically_complex_term_recall`).


Each dataset and each example comes with metrics colletion, accessed via the `metrics` attribute. Metric collections are responsible for ...


Bewer comes with a built-in metric registry that keeps track of all available metrics and their default configurations. Metrics are class-based, but may be registered under different names depending on the metric parameters and preprocessing pipeline. Each dataset has its own metric collection (`dataset.metrics`), which handles instantiation, parameterization, and caching of metrics. This means you never have to manage metric instances yourself — just request one by name, and the collection takes care of the rest.

**Metric values.** Every metric exposes a main value, typically `value` for numeric metrics and `alignment` for alignments, plus the constituent values, all accessible as attributes. For numeric metrics, a bootstrap confidence interval can be computed via `metric.compute_confidence_interval()`.




```python
# A metric and it's constituent values are exposed as attributes
metric_values = wer.metric_values()
assert "value" == metric_values["main"]
assert "num_edits" in metric_values["other"]
assert "ref_length" in metric_values["other"]

ci
```

```python

# Per-example access via iteration or indexing
ex_wer = wer[0]
assert ex_wer.value == ex_wer.num_edits / ex_wer.ref_length
```




### Key-term metrics

Key-term metrics measure how well specific terms (medical conditions, company names, acronyms, etc.) are recognized. To use them, build a [`Vocabulary`](src/bewer/core/vocabulary.py) and attach it to the dataset.

```python
# Add key terms from a Python list
key_terms = ["diabetes", "blood sugar"]
vocab = Vocabulary(name="key_terms").add_terms(key_terms)
dataset.add_vocabulary(vocab)

# Compute key-term metric by referencing the vocabulary name
print(dataset.metrics.ktr(vocab="key_terms").value)
```

You can also load line-separated key terms directly from a file (`add_file`) or write a custom vocabulary extractor, which is a callable `(dataset) -> Iterable[str]` that derives terms from the dataset's references (`add_extractor`).

### Alignments

Alignments produce an example-level text-to-text alignment as their primary output, rather than a dataset-level numeric score. An [`Alignment`](src/bewer/alignment/alignment.py) is a sequence of [`Op`](src/bewer/alignment/op.py) objects, each representing a match, substitution, insertion, or deletion between hypothesis and reference. Edit counts are available as supportive values:

```python
# Access alignments at the example level
example = dataset[0]
alignment = example.metrics.levenshtein().alignment

# Access pre-computed edit operation counts
assert alignment.num_edits + alignment.num_matches == len(alignment)
assert alignment.num_edits >= alignment.num_substitutions

# Print a color-coded two-row alignment in the console
alignment.display()
```
