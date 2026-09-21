
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

## Quickstart

```python
from bewer import Dataset

# Create an evaluation dataset
dataset = Dataset(language="en")

# Load data.
dataset.load_csv(
    "data.csv",
    ref_col="reference",
    hyp_col="hypothesis",
)

# List available metrics and compute.
dataset.metrics.list_metrics()
print(f"WER: {dataset.metrics.wer().value:.2%}")
```

<a name="metrics">

## Metrics

Metrics are computed lazily and cached, so requesting the same metric with the same parameters twice returns the cached result. Every metric exposes a **main value**, typically `.value` for numeric metrics and `.alignment` for alignments, plus optional **supportive values**, all accessible as attributes.

Requesting a metric *freezes* the dataset: its contents can no longer change, so further `add()`/`load_*()` calls raise `DatasetFrozenError`. Use `clone()` for a fresh, modifiable copy to keep building.

```python
wer = dataset.metrics.wer()          # returns a metric object (cached, freezes the dataset)

print(wer.metric_values())           # {'main': 'value', 'other': ['num_edits', 'ref_length']}

print(wer.value)                     # main value: 0.12
print(wer.num_edits)                 # supportive value: 15
print(wer.ref_length)                # supportive value: 125

# Per-example access via iteration or indexing:
example_metric = wer[0]
print(example_metric.value)
```

### Metrics catalog

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

### Alignment

Alignment metrics produce an example-level text-to-text alignment as their primary output, rather than a dataset-level numeric score. An [`Alignment`](src/bewer/alignment/alignment.py) is a sequence of [`Op`](src/bewer/alignment/op.py) objects, each representing a match, substitution, insertion, or deletion between hypothesis and reference. Edit counts are available as supportive values:

```python
# Access alignments at the example level
example = dataset[0]
alignment = example.metrics.levenshtein().alignment

# Access pre-computed edit operation counts
assert alignment.num_edits + alignment.num_matches == len(alignment)
assert alignment.num_edits >= alignment.num_substitutions

# Display a color-coded two-row alignment in the console
alignment.display()
```
