
# BeWER

*Beyond Word Error Rate → BeWER (/ˈbiːvər/) 🦫*

<p align="left">
  <img src="https://img.shields.io/badge/python-%203.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-green" alt="Python Versions">
  <img src="https://codecov.io/gh/corticph/bewer/graph/badge.svg?token=4QBH8TD4T4" alt="Coverage" style="margin-left:5px;">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" style="margin-left:5px;">
</p>

**⚠️ Important:** This project is not production ready and is still in early development. Breaking changes may occur, and backwards compatibility between alpha versions is not guaranteed.

**BeWER is an evaluation and analysis framework for automatic speech recognition in Python.** It defines a transparent YAML-based approach for configuring evaluation pipelines and makes it easy to inspect and analyze individual examples through a web-based interface. The built-in preprocessing pipeline and metrics collection are designed to cover all conventional use cases and then some, while still being fully extensible.




__Contents__ | [Installation](#installation) | [Quickstart](#quickstart) |


<a name="installation">

## Installation

```bash
pip install bewer
```

## Quickstart

**Create a Dataset**

```python
from bewer import Dataset

dataset = Dataset()
```

**Add data**

From a file:
```python
dataset.load_csv(
    "data.csv",
    ref_col="reference",
    hyp_col="hypothesis",
)
```

Or manually:
```python
for ref, hyp in iterator:
    dataset.add(ref=ref, hyp=hyp)
```

**List available metrics**

```python
dataset.metrics.list_metrics()
```

**Compute metrics lazily**

```python
print(f"WER: {dataset.metrics.wer().value:.2%}")
```
