from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from bewer.metrics.base import Metric

__all__ = ["ConfidenceInterval", "bootstrap_confidence_interval"]


@dataclass(frozen=True)
class ConfidenceInterval:
    """A confidence interval for a metric's main value.

    Iterable as ``(low, high)`` so ``low, high = ci`` works.
    """

    low: float
    high: float
    point: float
    level: float
    method: str
    n_resamples: int
    seed: Optional[int]

    def __iter__(self):
        yield self.low
        yield self.high

    @property
    def width(self) -> float:
        return self.high - self.low


def _percentile(sorted_samples: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy "linear" / type-7 convention)."""
    n = len(sorted_samples)
    if n == 1:
        return sorted_samples[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo])


def _all_value_attr_names(metric_cls) -> list[str]:
    mv = metric_cls.metric_values(include_private=True)
    names: list[str] = []
    if mv["main"]:
        names.append(mv["main"])
    names.extend(mv["other"])
    names.extend(mv["private"])
    return names


def _make_shadow(metric: "Metric", resampled_src: list) -> "Metric":
    """Build a throwaway Metric instance whose `_src` is a resampled list of
    examples. Shares the original's `_examples` cache (and its dependency
    metrics' caches) so per-example contributions are not recomputed.
    """
    shadow = type(metric).__new__(type(metric))
    shadow.__dict__.update(metric.__dict__)
    shadow._src = resampled_src
    for dep_name in type(metric).dependencies():
        if dep_name in metric.__dict__:
            shadow.__dict__[dep_name] = _make_shadow(metric.__dict__[dep_name], resampled_src)
    for vname in _all_value_attr_names(type(metric)):
        shadow.__dict__.pop(vname, None)
    return shadow


def _warm_caches(metric: "Metric") -> None:
    """Force computation of the main value and all dependency main values once,
    populating the per-example caches that bootstrap will reuse.
    """
    main_name = type(metric).metric_values()["main"]
    if main_name is not None:
        getattr(metric, main_name)
    for dep_name in type(metric).dependencies():
        dep = getattr(metric, dep_name)
        _warm_caches(dep)


def bootstrap_confidence_interval(
    metric: "Metric",
    level: float = 0.95,
    n_resamples: int = 1000,
    seed: Optional[int] = None,
) -> ConfidenceInterval:
    """Compute a percentile bootstrap CI for ``metric``'s main value by
    resampling examples (with replacement) and recomputing the metric on each
    resample. Works generically for any Metric subclass whose main value is a
    scalar.
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    main_name = type(metric).metric_values()["main"]
    if main_name is None:
        raise ValueError(f"{type(metric).__name__} has no main metric value; cannot compute a confidence interval.")

    examples = list(metric._src)
    n = len(examples)
    if n == 0:
        raise ValueError("Cannot compute a confidence interval on an empty dataset.")

    _warm_caches(metric)
    point = getattr(metric, main_name)
    if not isinstance(point, (int, float)) or isinstance(point, bool):
        raise TypeError(
            f"compute_confidence_interval requires a numeric metric value; "
            f"{type(metric).__name__}.{main_name} returned {type(point).__name__}."
        )

    if n == 1:
        return ConfidenceInterval(
            low=float(point),
            high=float(point),
            point=float(point),
            level=level,
            method="bootstrap",
            n_resamples=n_resamples,
            seed=seed,
        )

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        resampled = rng.choices(examples, k=n)
        shadow = _make_shadow(metric, resampled)
        samples.append(float(getattr(shadow, main_name)))

    samples.sort()
    alpha = (1.0 - level) / 2.0
    return ConfidenceInterval(
        low=_percentile(samples, alpha),
        high=_percentile(samples, 1.0 - alpha),
        point=float(point),
        level=level,
        method="bootstrap",
        n_resamples=n_resamples,
        seed=seed,
    )
