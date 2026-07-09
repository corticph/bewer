#!/usr/bin/env python
"""Metric regression harness for bewer.

Computes a fixed set of metric values over committed datasets and compares them
to committed "golden" baselines, so that a release cannot silently change the
numbers a metric produces for a fixed input.

Usage:
    # Check current metrics against committed baselines (exit 1 on drift).
    python regression/runner.py

    # Regenerate baselines after a *sanctioned* metric change (explain why in the commit).
    python regression/runner.py --update

    # Limit to a single dataset.
    python regression/runner.py --dataset earnings21

The set of datasets and the exact metric values to guard are declared in
regression/manifest.yml. See that file for the schema.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

import bewer
from bewer import Dataset, Vocabulary

# runner.py lives in <repo>/regression/, so the repo root is its parent's parent.
REGRESSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = REGRESSION_DIR.parent
DEFAULT_MANIFEST = REGRESSION_DIR / "manifest.yml"
BASELINES_DIR = REGRESSION_DIR / "baselines"

console = Console()

# One computed metric entry, keyed in a dataset's `metrics` by its manifest name:
#   {"params": {param_name: value}, "values": {value_name: number}}
# `params` are the fully-resolved metric params (schema fields + pipeline), tracked so a
# change to a metric's *default* (e.g. normalized, tokenizer) fails the check explicitly.
MetricEntry = dict[str, dict[str, Any]]


@dataclass
class Diff:
    """A single mismatch between a computed entry and its baseline."""

    dataset: str
    metric: str
    field: str  # value or param name (params prefixed "param:" in the report)
    baseline: Any
    computed: Any
    kind: str  # "value" | "param" | "missing" (in computed) | "new" (not in baseline)


# --------------------------------------------------------------------------------------
# Manifest / paths
# --------------------------------------------------------------------------------------


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """Load and return the parsed regression manifest."""
    with open(manifest_path) as f:
        return yaml.safe_load(f)


def _resolve(path: str) -> Path:
    """Resolve a manifest path (relative to the repo root)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def baseline_path(dataset_name: str) -> Path:
    return BASELINES_DIR / f"{dataset_name}.yml"


# --------------------------------------------------------------------------------------
# Computation
# --------------------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    # bool is a subclass of int; allow it (it round-trips through YAML cleanly).
    return isinstance(value, (int, float)) and not isinstance(value, complex)


def _resolve_value(metric: Any, value_name: str) -> int | float:
    """Resolve a single named metric value to a number.

    1. If the dataset-level metric exposes the value as a scalar, use it directly
       (e.g. WER.value / WER.num_edits).
    2. Otherwise read it from each example metric and sum across examples. This is
       valid because example-only ``metric_value``s in bewer are additive counts
       (e.g. Levenshtein.num_substitutions).
    3. If it is neither, raise — catches typos and non-numeric values such as the
       Levenshtein ``alignment`` object.
    """
    if hasattr(metric, value_name):
        value = getattr(metric, value_name)
        if _is_number(value):
            return value
        # Present but non-numeric (e.g. an Alignment object): not trackable.
        raise TypeError(
            f"Metric '{metric.short_name}' value '{value_name}' is "
            f"{type(value).__name__}, not a number. Choose a numeric value to track."
        )

    # Fall back to aggregating the example-level value across examples.
    try:
        example_values = [getattr(example_metric, value_name) for example_metric in metric]
    except (AttributeError, TypeError) as e:
        raise AttributeError(
            f"Metric '{metric.short_name}' has no value '{value_name}' at the dataset "
            f"or example level. Check the `values:` list in the manifest."
        ) from e

    if not all(_is_number(v) for v in example_values):
        raise TypeError(
            f"Example-level value '{value_name}' on metric '{metric.short_name}' is not "
            f"numeric, so it cannot be aggregated for regression tracking."
        )
    return sum(example_values)


def build_dataset(dataset_spec: dict) -> Dataset:
    """Build a fresh, uncomputed Dataset (data loaded, vocab attached) for one spec.

    Datasets are always committed as JSONL with ``ref``/``hyp`` columns (we control the
    format), so there is nothing to configure per dataset beyond the path and language.
    """
    language = dataset_spec.get("language")
    ds = Dataset(language=language) if language else Dataset()
    ds.load_jsonl(str(_resolve(dataset_spec["path"])), ref_col="ref", hyp_col="hyp")

    vocab_spec = dataset_spec.get("vocab")
    if vocab_spec:
        vocab = Vocabulary(name=vocab_spec["name"]).add_file(_resolve(vocab_spec["file"]))
        ds.add_vocabulary(vocab)
    return ds


def _metric_params(metric: Any) -> dict[str, Any]:
    """Fully-resolved params for a metric: schema fields + pipeline (all plain scalars).

    Tracking these lets the check fail explicitly if a metric's *default* changes
    (e.g. a param default flips, or a registered tokenizer/normalizer changes).
    """
    params: dict[str, Any] = {}
    if metric.params is not None:
        for f in fields(metric.params):
            params[f.name] = getattr(metric.params, f.name)
    params["standardizer"] = metric.standardizer
    params["tokenizer"] = metric.tokenizer
    params["normalizer"] = metric.normalizer
    return params


def _metric_keys(metric_specs: list[dict]) -> list[str]:
    """Stable per-metric keys = manifest name, disambiguated only on collision.

    Keying by the manifest name (not the resolved short_name) keeps the key stable when
    a default changes, so the change surfaces as a clear param diff rather than as a
    new/missing key. When a name repeats in one dataset it is disambiguated by its explicit
    params, and an occurrence suffix (``#2``) is appended if that is still not unique, so
    the returned keys are always distinct.
    """
    counts: dict[str, int] = {}
    for spec in metric_specs:
        counts[spec["name"]] = counts.get(spec["name"], 0) + 1
    keys: list[str] = []
    seen: dict[str, int] = {}
    for spec in metric_specs:
        name = spec["name"]
        key = name
        if counts[name] > 1 and spec.get("params"):
            suffix = ",".join(f"{k}={v}" for k, v in sorted(spec["params"].items()))
            key = f"{name}[{suffix}]"
        # Guarantee uniqueness even if the (possibly param-less) key still collides.
        n = seen.get(key, 0)
        seen[key] = n + 1
        keys.append(key if n == 0 else f"{key}#{n + 1}")
    return keys


def compute(
    dataset_spec: dict, *, isolate: bool = True, progress: bool = False
) -> tuple[dict[str, MetricEntry], dict[str, float]]:
    """Compute every configured metric and return (entries, per-metric wall-clock seconds).

    With ``isolate=True`` (default) each metric is computed on its OWN freshly built
    dataset, so a metric that shares an underlying computation with another (e.g. KTCER
    and RKTR both build the error_align alignment; the KT-family shares _kt_stats) pays
    its full cost rather than being deflated by a warm cache left behind by a sibling.
    The timing then reflects each metric's true isolated cost (incl. lazy preprocessing
    and shared dependencies) from a cold state.

    With ``isolate=False`` all metrics are computed on a single shared dataset (caches
    reused across metrics). This is faster and gives identical *values*, but the timing
    is deflated for cache-sharing metrics — use it when only the values matter (e.g. the
    pytest value check), not when reporting per-metric time.

    ``progress=True`` shows a live per-metric progress bar (metrics are computed serially
    and some take tens of seconds, so a silent run looks hung).
    """
    entries: dict[str, MetricEntry] = {}
    timing: dict[str, float] = {}
    shared_ds = None if isolate else build_dataset(dataset_spec)
    metric_specs = dataset_spec["metrics"]
    keys = _metric_keys(metric_specs)

    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]
    progress_bar = Progress(*columns, console=console, transient=True, disable=not progress)
    with progress_bar:
        task = progress_bar.add_task(f"{dataset_spec['name']}", total=len(metric_specs))
        for key, metric_spec in zip(keys, metric_specs):
            name = metric_spec["name"]
            params = metric_spec.get("params", {})
            value_names = metric_spec["values"]

            progress_bar.update(task, description=f"{dataset_spec['name']}: computing [cyan]{key}[/cyan]")
            ds = build_dataset(dataset_spec) if isolate else shared_ds
            start = time.perf_counter()
            metric = ds.metrics.get(name)(**params)
            values = {vn: _resolve_value(metric, vn) for vn in value_names}
            elapsed = time.perf_counter() - start

            entries[key] = {"params": _metric_params(metric), "values": values}
            timing[key] = elapsed
            progress_bar.advance(task)

    return entries, timing


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


def _values_match(baseline: Any, computed: Any, tolerance: float) -> bool:
    if isinstance(baseline, int) and isinstance(computed, int):
        return baseline == computed
    return abs(float(baseline) - float(computed)) <= tolerance


def compare(dataset_name: str, computed: dict[str, MetricEntry], baseline: dict, tolerance: float) -> list[Diff]:
    """Compare computed entries (params + values) against a baseline's ``metrics`` section.

    Values are compared within ``tolerance`` (ints exactly); params are compared exactly,
    so a change to a metric's resolved defaults (e.g. ``normalized``, ``tokenizer``) is
    flagged as a ``param`` diff rather than silently altering values.
    """
    diffs: list[Diff] = []
    baseline_metrics = baseline.get("metrics", {})

    for key, entry in computed.items():
        if key not in baseline_metrics:
            for vn, cv in entry["values"].items():
                diffs.append(Diff(dataset_name, key, vn, None, cv, "new"))
            continue
        base = baseline_metrics[key]

        # Params: exact comparison (default drift detection).
        base_params = base.get("params", {})
        for pn, cv in entry["params"].items():
            if pn not in base_params:
                diffs.append(Diff(dataset_name, key, f"param:{pn}", None, cv, "new"))
            elif base_params[pn] != cv:
                diffs.append(Diff(dataset_name, key, f"param:{pn}", base_params[pn], cv, "param"))
        for pn, bv in base_params.items():
            if pn not in entry["params"]:
                diffs.append(Diff(dataset_name, key, f"param:{pn}", bv, None, "missing"))

        # Values: tolerance comparison.
        base_values = base.get("values", {})
        for vn, cv in entry["values"].items():
            if vn not in base_values:
                diffs.append(Diff(dataset_name, key, vn, None, cv, "new"))
            elif not _values_match(base_values[vn], cv, tolerance):
                diffs.append(Diff(dataset_name, key, vn, base_values[vn], cv, "value"))
        for vn, bv in base_values.items():
            if vn not in entry["values"]:
                diffs.append(Diff(dataset_name, key, vn, bv, None, "missing"))

    # Baseline metrics no longer produced at all.
    for key in baseline_metrics:
        if key not in computed:
            diffs.append(Diff(dataset_name, key, "*", "present", None, "missing"))

    return diffs


# --------------------------------------------------------------------------------------
# Baseline IO
# --------------------------------------------------------------------------------------


def write_baseline(dataset_name: str, computed: dict[str, MetricEntry], timing: dict[str, float]) -> None:
    """Write/overwrite a dataset's YAML baseline, stamped with the bewer version.

    Who/when/why a baseline changed lives in git (commit + PR), so it is not
    duplicated here. Only ``bewer_version`` is kept — git cannot tell you which
    bewer produced the numbers.

    ``compute_seconds`` (per-metric wall-clock) is recorded for reference only; it is
    NOT part of the regression comparison since wall-clock time is not deterministic.
    """
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    # Nest the (informational, not-compared) per-metric compute time inside each entry.
    metrics = {key: {**entry, "compute_seconds": round(timing[key], 3)} for key, entry in computed.items()}
    payload = {
        "bewer_version": bewer.__version__,
        "metrics": metrics,
    }
    with open(baseline_path(dataset_name), "w") as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False, allow_unicode=True)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _render_diffs(diffs: list[Diff]) -> None:
    table = Table(title="Metric regression drift", show_lines=False)
    table.add_column("Dataset")
    table.add_column("Metric")
    table.add_column("Field")
    table.add_column("Baseline", justify="right")
    table.add_column("Computed", justify="right")
    table.add_column("Kind")
    for d in diffs:
        table.add_row(
            d.dataset,
            d.metric,
            d.field,
            "-" if d.baseline is None else f"{d.baseline}",
            "-" if d.computed is None else f"{d.computed}",
            d.kind,
        )
    console.print(table)


# Timing is measured cold per metric and is inherently noisy; only color a Δ time swing
# beyond this magnitude (percent) so a red/green cell actually signals something.
TIME_DEADBAND_PCT = 10.0


def _fmt_value_delta(computed: dict, base_values: dict, tolerance: float) -> str:
    """Signed delta of the metric's main ``value``: green if within tolerance, else red."""
    if "value" not in computed.get("values", {}):
        return "-"  # metric tracks no main value (e.g. Levenshtein)
    if "value" not in base_values:
        return "-"  # no baseline to compare (update mode / new metric)
    cur, base = computed["values"]["value"], base_values["value"]
    if _values_match(base, cur, tolerance):
        return "[green]≈0[/green]"
    return f"[red]{cur - base:+.4g}[/red]"


def _fmt_time_delta(secs: float, base_secs: float | None) -> str:
    """Percent change vs baseline; green if faster, red if slower, uncolored within band."""
    if base_secs is None or base_secs == 0:
        return "-"
    pct = (secs - base_secs) / base_secs * 100
    text = f"{pct:+.1f}%"
    if abs(pct) < TIME_DEADBAND_PCT:
        return f"[dim]{text}[/dim]"
    return f"[red]{text}[/red]" if pct > 0 else f"[green]{text}[/green]"


def _fmt_values_passed(computed: dict, base_values: dict, have_baseline: bool, tolerance: float) -> str:
    """N/M tracked values matching baseline within tolerance; green iff all pass."""
    if not have_baseline:
        return "-"
    values = computed.get("values", {})
    total = len(values)
    passed = sum(1 for vn, cv in values.items() if vn in base_values and _values_match(base_values[vn], cv, tolerance))
    color = "green" if passed == total else "red"
    return f"[{color}]{passed}/{total}[/{color}]"


def _render_summary(
    dataset_name: str,
    computed: dict[str, MetricEntry],
    timing: dict[str, float],
    baseline: dict | None,
    tolerance: float,
) -> None:
    """Per-metric summary: main value + deltas, isolated compute time + delta, pass count."""
    baseline_metrics = (baseline or {}).get("metrics", {})
    table = Table(title=f"Regression summary — {dataset_name}", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Δ value", justify="right")
    table.add_column("Values", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Δ time", justify="right")

    total_passed = total_values = 0
    # Slowest-first so the expensive metrics stay at the top.
    for key, secs in sorted(timing.items(), key=lambda kv: kv[1], reverse=True):
        entry = computed[key]
        base = baseline_metrics.get(key)
        base_values = base.get("values", {}) if base else {}
        base_secs = base.get("compute_seconds") if base else None

        main = entry["values"].get("value")
        value_str = f"{main:.4f}" if isinstance(main, float) else ("-" if main is None else str(main))
        if base is not None:
            total_values += len(entry["values"])
            total_passed += sum(
                1
                for vn, cv in entry["values"].items()
                if vn in base_values and _values_match(base_values[vn], cv, tolerance)
            )
        table.add_row(
            key,
            value_str,
            _fmt_value_delta(entry, base_values, tolerance),
            _fmt_values_passed(entry, base_values, base is not None, tolerance),
            f"{secs:.3f}",
            _fmt_time_delta(secs, base_secs),
        )

    total_pass_str = f"{total_passed}/{total_values}" if baseline_metrics else ""
    table.add_section()  # horizontal rule before the total row
    table.add_row(
        "[bold]total[/bold]",
        "",
        "",
        f"[bold]{total_pass_str}[/bold]",
        f"[bold]{sum(timing.values()):.3f}[/bold]",
        "",
    )
    console.print()  # blank line so consecutive tables don't melt together
    console.print(table)


def _print_legend() -> None:
    """Print the summary-table legend once (shared across all datasets' tables)."""
    # highlight=False disables rich's auto-highlighter, which would otherwise color the
    # "10" in ±10% and the "/" in "value / Values". Explicit [green]/[red] markup still applies.
    console.print()
    console.print("Legend:", highlight=False)
    console.print("  • Compute time is measured per metric in isolation from a cold cache.", highlight=False)
    console.print(
        f"  • Δ time colored beyond ±{TIME_DEADBAND_PCT:.0f}%: [green]green[/green] = faster, [red]red[/red] = slower.",
        highlight=False,
    )
    console.print(
        "  • Δ value / Values: [green]green[/green] = within tolerance, [red]red[/red] = drift.", highlight=False
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bewer metric regression check.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to the manifest.")
    parser.add_argument("--update", action="store_true", help="Regenerate baselines instead of checking.")
    parser.add_argument("--dataset", default=None, help="Limit to a single dataset by name.")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    tolerance = float(manifest.get("tolerance", 1e-9))
    # Live progress bar only on an interactive terminal; keeps CI logs quiet.
    progress = sys.stdout.isatty()
    datasets = manifest["datasets"]
    if args.dataset:
        datasets = [d for d in datasets if d["name"] == args.dataset]
        if not datasets:
            console.print(f"[red]No dataset named '{args.dataset}' in manifest.[/red]")
            return 2

    if args.update:
        for spec in datasets:
            computed, timing = compute(spec, progress=progress)
            write_baseline(spec["name"], computed, timing)
            console.print(f"[green]Updated baseline:[/green] {baseline_path(spec['name']).relative_to(REPO_ROOT)}")
            _render_summary(spec["name"], computed, timing, None, tolerance)
        _print_legend()
        return 0

    all_diffs: list[Diff] = []
    missing_baselines: list[str] = []
    rendered_any = False
    for spec in datasets:
        name = spec["name"]
        bpath = baseline_path(name)
        if not bpath.is_file():
            missing_baselines.append(name)
            continue
        with open(bpath) as f:
            baseline = yaml.safe_load(f)
        computed, timing = compute(spec, progress=progress)
        all_diffs.extend(compare(name, computed, baseline, tolerance))
        _render_summary(name, computed, timing, baseline, tolerance)
        rendered_any = True

    if rendered_any:
        _print_legend()

    if missing_baselines:
        console.print(
            f"[red]Missing baseline file(s) for: {', '.join(missing_baselines)}. "
            f"Run with --update to create them.[/red]"
        )

    if all_diffs:
        _render_diffs(all_diffs)
        console.print(
            f"[red]Metric regression FAILED: {len(all_diffs)} difference(s) from baseline.[/red] "
            "If this change is intended, run the Metric Regression workflow in update mode "
            "(or `make update-baselines`) to register the new values."
        )

    if missing_baselines or all_diffs:
        return 1

    console.print()
    console.print("[green]Metric regression passed: all tracked values match their baselines.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
