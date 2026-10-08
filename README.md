
<img src="https://raw.githubusercontent.com/corticph/bewer/main/.github/assets/logo.svg" alt="BeWER" width="100%"/>

<p align="center">
  <img src="https://img.shields.io/badge/python-%203.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-green" alt="Python Versions">
  <img src="https://codecov.io/gh/corticph/bewer/graph/badge.svg?token=4QBH8TD4T4" alt="Coverage" style="margin-left:5px;">
  <img src="https://img.shields.io/pypi/v/bewer" alt="PyPI">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" style="margin-left:5px;">
</p>

<br>

**⚠️ Important:** Bewer is pre-1.0 and under active development. The API may change. Breaking changes are signalled by a **minor** version bump (e.g. `0.2.x → 0.3.0`); patch releases contain only fixes and backwards-compatible additions. Pin with `bewer~=0.Y.0` (pip) or `^0.Y.0` (Poetry).

**Bewer is an evaluation and analysis framework for automatic speech recognition in Python.** It defines a flexible approach for configuring and customizing speech recognition evaluation. The hierarchical structure, going from the dataset level down to individual tokens, makes it easy to inspect individual examples and understand evaluation results. The built-in preprocessing pipeline and metrics collection are designed to cover all conventional use cases and then some, while still being fully extensible.

__Contents__ | [Installation](#installation) | [Quickstart](#quickstart) | [Core Concepts](#core-concepts) | [Metrics Catalog](#metrics-catalog) |

## Installation

```bash
pip install bewer
```

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

# Compute a metric
wer = dataset.metrics.wer()
print(f"{wer.short_name_base}: {wer.value:.2%}")
```

```text
WER: 12.34%
```

## Core Concepts

### Hierarchy

In `bewer`, evaluation is centered around the `Dataset` object, which implements a hierarchical structure, from a collection of reference-hypothesis pairs to individual tokens.

```python
dataset = Dataset(language="en")            # Create a dataset ...
dataset.add(ref="one two", hyp="want to")   # ... and populate it

# Climb down the hierarchy:
for example in dataset:                     # Dataset -> Example
    print(example)
    text = example.ref                      # Example -> Text
    print(text)
    for token in text.tokens:               # Text -> Token
        print(token)
```

```text
Example(ref="one two", hyp="want to")
Text("one two")
Token("one")
Token("two")
```

### Preprocessing

Text preprocessing is a central part of speech recognition evaluation, and typically requires specific considerations for different languages, domains, or tasks. The `bewer` preprocessing pipeline runs in three stages:

- **Standardization** runs on the whole string, before tokenization. It irons out inconsistencies that would otherwise affect how the text is split. The default standardizer applies Unicode NFC and unifies apostrophe, hyphen, and slash variants.
- **Tokenization** is regex-based, to track token boundaries in the standardized text. While not essential for metric computation, it supports richer post-hoc analysis.
- **Normalization** runs per token, after tokenization. At this point, token boundaries are fixed. It typically handles lowercasing, removing diacritics, and other language-specific adjustments.

The result of each stage is accessible from the corresponding objects.

```python
text.raw                      # str: the original input
text.standardized             # str: after the standardizer
text.tokens.standardized      # list[str]: the tokens after standardization
text.tokens.normalized        # list[str]: the tokens after normalization
```

Pre-defined pipeline components can be found under `dataset.pipelines`.

```python
print(dataset.pipelines)
```

```text
Pipelines(
    standardizers: default
    tokenizers:    default, with_punctuation, key_term, orthographically_complex_term
    normalizers:   default, cased
)
```

Each component is named and defined as part of the dataset's configuration. Typically, it is sufficient to just choose the language, and the configuration for that language is applied automatically, but you can adapt and extend the pipeline as needed.

```python
# TODO: Example of customizing the pipeline for a dataset
```

Any component can be switched with the `set_pipeline` context manager. The English `key_term` tokenizer, for instance, splits on apostrophes as well as hyphens and slashes, so that a term is still matched when it appears in the possessive form. In contrast, the `default` tokenizer does not.

```python
from bewer import set_pipeline

dataset = Dataset(language="en")
dataset.add(ref="Jane Doe's", hyp="Jane Doe's")
text = dataset[0].ref

print(text.tokens.standardized)

with set_pipeline(tokenizer="key_term"):
    print(text.tokens.standardized)
```

```text
['Jane', "Doe's"]
['Jane', 'Doe', 's']
```

In practice you will rarely need to adjust the pipeline manually. As we will see next, metrics carry the pipeline they run under, so a key-term metric is already defined to use the `key_term` tokenizer.

### Metrics

Every metric in `bewer` is registered under one or more names, which makes it accessible from a dataset's metrics collection.

```python
wer = dataset.metrics.wer()
```

All metrics accept the preprocessing components `standardizer`, `tokenizer` and `normalizer` as keyword arguments, allowing you to override the defaults they were registered with. Some metrics also accept arguments specific to their computation.

```python
# Override the pipeline components a metric runs under (see Vocabularies below
# for how to define a key-term vocabulary):
ktf = dataset.metrics.ktf(
    tokenizer="key_term",
    vocab="medical",
)
```

Call `dataset.metrics.list_metrics()` to get an overview of available metrics and their parameters or see the [metrics catalog](#metrics-catalog) below.

You can also register new variants of existing metrics or your own custom metrics.

```python
from bewer.metrics import METRIC_REGISTRY, WER

# Register a custom metric in the global registry
METRIC_REGISTRY.register_metric(WER, name="my_wer", tokenizer="key_term")

# Access it from the dataset's metrics collection
my_wer = dataset.metrics.my_wer()
```

Every metric exposes a main value alongside the constituents it was computed from. By convention, the main value is simply `value` for numeric metrics and `alignment` for alignments.

```python
print(wer.metric_values())
print(f"{wer.value:.2%} = {wer.num_edits}/{wer.ref_length}")
```

```text
{'main': 'value', 'other': ['num_edits', 'ref_length']}
12.50% = 1/8
```

Most metrics are defined by aggregating over example-level values. This structure is reflected in `bewer`, which also exposes example-level metrics accessible through the example objects or directly from the metric object itself.

```python
wer = dataset.metrics.wer()

assert wer[0] is dataset[0].metrics.wer()
assert wer.num_edits == sum(ex_wer.num_edits for ex_wer in wer)
```

#### Vocabularies

Key-term metrics measure performance on a pre-defined subset of terms (medical, financial, acronyms, etc.). To use them, build a [`Vocabulary`](src/bewer/core/vocabulary.py) and attach it to the dataset.

```python
from bewer import Vocabulary

dataset = Dataset(language="en")
dataset.add(
    ref="the patient has diabetes and high blood sugar",
    hyp="the patient has diabetis and high blood sugar",
)

# Add key terms from a Python list
vocab = Vocabulary(name="medical")
vocab.add_terms(["diabetes", "blood sugar"])
dataset.add_vocabulary(vocab)

# Compute key-term recall by referencing the vocabulary name
ktr = dataset.metrics.ktr(vocab="medical")
print(f"{ktr.long_name_base}: {ktr.value:.2%}")
```

```text
Key-Term Recall: 50.00%
```

You can also load line-separated key terms directly from a file (`add_file`) or write a custom function (`Dataset -> Iterable[str]`) to extract key terms from the dataset (`add_extractor`).

#### Alignments

Alignments produce an example-level text-to-text alignment as their primary output, rather than a dataset-level numeric score. An [`Alignment`](src/bewer/alignment/alignment.py) is a sequence of [`Op`](src/bewer/alignment/op.py) objects, each representing a match, substitution, insertion, or deletion between hypothesis and reference. Edit counts are available as supportive values.

```python
dataset = Dataset(language="en")
dataset.add(
    ref="an example with different types of errors",
    hyp="an odd example with diff types errors",
)

# Access alignments at the example level
example = dataset[0]
alignment = example.metrics.levenshtein().alignment

# Access pre-computed edit operation counts
assert alignment.num_edits + alignment.num_matches == len(alignment)
assert alignment.num_edits >= alignment.num_substitutions

# Print a color-coded two-row alignment in the console
alignment.display()
```

<img src="https://raw.githubusercontent.com/corticph/bewer/main/.github/assets/alignment-display.png" alt="A color-coded two-row alignment, showing an insertion, a substitution and a deletion" width="100%"/>

### Lazy Computation and Caching

Metric values and pipeline attributes are computed lazily. The dependencies between the pipeline stages are tracked to make sure that the full pipeline is considered when deciding which computations are necessary. Once a metric is requested from the metrics collection, it is cached and reused for subsequent requests with matching parameters or by other metrics that depend on it.

```python
wer = dataset.metrics.wer() # already uses the "default" normalizer
assert wer is dataset.metrics.wer(normalizer="default")
```

## Metrics Catalog

| | Type | Accessor | |
|--------|------|----------|---:|
| **General purpose** | | | |
| Word Error Rate | General | `wer` | [`>`](src/bewer/metrics/wer.py) |
| Character Error Rate | General | `cer` | [`>`](src/bewer/metrics/cer.py) |
| Punctuation Error Rate | General | `per` | [`>`](src/bewer/metrics/per.py) |
| **Hallucination metrics** | | | |
| Insertion Rate | Hallucination | `insertion_rate` | [`>`](src/bewer/metrics/insertion_rate.py) |
| **Key-term metrics** | | | |
| Key-Term Recall | Key-term | `ktr` | [`>`](src/bewer/metrics/ktr.py) |
| Key-Term Precision | Key-term | `ktp` | [`>`](src/bewer/metrics/ktp.py) |
| Key-Term F-Score | Key-term | `ktf` | [`>`](src/bewer/metrics/ktf.py) |
| Key-Term Error Rate | Key-term | `kter` | [`>`](src/bewer/metrics/kter.py) |
| Key-Term False-Positive Rate | Key-term | `ktfpr` | [`>`](src/bewer/metrics/ktfpr.py) |
| Key-Term Character Error Rate | Key-term | `ktcer` | [`>`](src/bewer/metrics/ktcer.py) |
| Relaxed Key-Term Recall | Key-term | `rktr` | [`>`](src/bewer/metrics/rktr.py) |
| **Orthographically complex terms** | | | |
| Orthographically Complex Term Recall | Key-term | `orthographically_complex_term_recall` | [`>`](src/bewer/metrics/orthographically_complex_term.py) |
| Orthographically Complex Term Precision | Key-term | `orthographically_complex_term_precision` | [`>`](src/bewer/metrics/orthographically_complex_term.py) |
| Orthographically Complex Term F-Score | Key-term | `orthographically_complex_term_fscore` | [`>`](src/bewer/metrics/orthographically_complex_term.py) |
| **Alignments** | | | |
| Levenshtein Alignment | Alignment | `levenshtein` | [`>`](src/bewer/metrics/levenshtein.py) |
| Error Alignment | Alignment | `error_align` | [`>`](src/bewer/metrics/error_align.py) |
| **Dataset statistics** | | | |
| Dataset Summary | Summary | `summary` | [`>`](src/bewer/metrics/summary.py) |
