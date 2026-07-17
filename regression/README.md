# Metric regression

Guards against unintended changes to bewer's metric values between releases. The
runner computes a fixed set of metrics over committed datasets and compares them to
committed "golden" baselines. On a version tag (and on manual dispatch) the workflow
fails if anything drifts — see
[`../.github/workflows/metric_regression.yml`](../.github/workflows/metric_regression.yml).

## Layout

- `manifest.yml` — datasets and the exact metric values guarded per dataset.
- `datasets/<name>/data.jsonl` — committed refs + hyps (the frozen fixtures).
- `datasets/<name>/generate/` — self-contained poetry project that rebuilds that
  dataset's `data.jsonl`; its `poetry.lock` is the provenance record.
- `baselines/<name>.yml` — committed golden values (with params and compute time).
- `runner.py` — computes metrics and checks or updates the baselines.

## Run the check

```bash
make regression        # or: poetry run python regression/runner.py
```

Compares current metrics to the baselines and exits non-zero on drift. Add
`--dataset <name>` to scope to one dataset.

## Update baselines (sanctioned change)

```bash
make update-baselines  # or: poetry run python regression/runner.py --update
```

Regenerates the baseline files. Commit them **in the same PR** as the metric change
(explain why in the commit/PR) so the check passes and the reviewer sees both together.

## Regenerate the `data.jsonl`

Each dataset has its own isolated generation env (heavy ML deps, kept out of bewer's
own environment). To rebuild one dataset's `data.jsonl`:

```bash
cd datasets/<name>/generate
poetry install
poetry run python generate.py          # see `--help` for options (e.g. --device)
```

The `poetry.lock` pins the exact stack and the model weights are pinned in
`generate.py`. GPU/CUDA are not pinned, but the committed `data.jsonl` is the frozen
fixture the check uses, so bit-exact regeneration isn't required to run the regression.
