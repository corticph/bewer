
<span style="font-size:32px; font-weight:700; line-height:1;">
BeWER
</span>

---

<span style="font-style:italic; white-space:nowrap; line-height:1;">
Beyond Word Error Rate → BeWER (/ˈbiːvər/) 🦫
</span>


---

**BeWER is an evaluation and analysis framework for automatic speech recognition in Python.** It defines a transparent YAML-based approach for configuring evaluation pipelines and makes it easy to inspect and analyze individual examples through a web-based interface. The built-in preprocessing pipeline and metrics collection are designed to cover all conventional use cases and then some, while still being fully extensible.




__Contents__ | [Installation](#installation) | [Quickstart](#quickstart) |


<a name="installation">

## Installation

For development:
```bash
make install
```

As a dependency:
```toml
bewer = { git = "ssh://git@github.com/corticph/bewer.git", tag="v0.1.0a1"}
```

## Quickstart

Instantiate a Dataset object:
```python
from bewer.core import Dataset

dataset = Dataset()
```

Add evaluation data from a file:
```python
dataset.load_csv(
    "data.csv",
    ref_col="gold_transcripts",
    hyp_col="model_transcripts",
)
```

Add evaluation examples manually:
```python
for ref, hyp in iterator:
    dataset.add(ref=ref, hyp=hyp)
```
