<div style="display: flex; justify-content: space-between; align-items: center;">
  <h1 style="margin: 0;">BeWER</h1>
  <span style="font-size: 16px; font-style: italic;">
    Beyond Word Error Rate &nbsp; → &nbsp; BeWER &nbsp; (/ˈbiːvər/) &nbsp; 🦫
  </span>
</div>

<br/>

---

**BeWER is an evaluation and analysis framework for automatic speech recognition in Python.** It defines a transparent YAML-based approach for configuring evaluation pipelines and makes it easy to inspect and analyze individual examples through a web-based interface. The built-in preprocessing pipeline and metrics collection are designed to cover all conventional use cases and then some, while still being fully extensible.




__Contents__ | [Installation](#installation) | [Quickstart](#quickstart) |

Template for creating Python packages. Put your short project description here.

To configure this package and create an Azure Pipeline, see [this Notion page](https://www.notion.so/cortihome/Creating-a-new-GitHub-repository-with-CI-pipeline-9241fb356ead448b941a9d4cfa4daf73).

## Installation

### For development purposes

```bash
make install
```

## Run tests

```bash
make test
make pre-commit
```

### As a dependency

Add the following line to your `pyproject.toml` file:

```toml
python-package-template = { git = "ssh://git@github.com/corticph/python-package-template.git", tag="vX.Y.Z"}
```
