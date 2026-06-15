"""Quantity metrics — predefined regex metrics for numbers and number+unit/symbol spans.

Each category is its own flat, standalone regex metric (registered via
:func:`~bewer.metrics.regex_metrics.register_regex_metric`): ``number``, ``percentage``,
``degree``, ``currency`` and ``measurement``. They are siblings of the complex-term metric and
of any metric a user registers — none is special, and there is intentionally no combined
"quantity" metric; they merely share being defined by a regular expression::

    dataset.metrics.number_recall().value
    dataset.metrics.percentage_precision().value
    dataset.metrics.measurement_recall().value

They run under the ``orthographically_complex_term`` tokenizer (which keeps units/symbols as their own tokens)
and the ``cased`` normalizer (which does not lowercase), so a quantity's surface form —
including unit case (mg vs Mg, MHz vs mHz) — is scored strictly. Categories may overlap:
``number`` matches every number, including those inside a percentage/currency/measurement.
"""

from __future__ import annotations

from bewer.extractors.quantity import (
    CURRENCY_PATTERN,
    DEGREE_PATTERN,
    MEASUREMENT_PATTERN,
    NUMBER_PATTERN,
    PERCENTAGE_PATTERN,
)
from bewer.metrics.regex_metrics import register_regex_metric

NumberRecall, NumberPrecision, NumberFscore = register_regex_metric("number", NUMBER_PATTERN, span=False)
PercentageRecall, PercentagePrecision, PercentageFscore = register_regex_metric(
    "percentage", PERCENTAGE_PATTERN, span=True
)
DegreeRecall, DegreePrecision, DegreeFscore = register_regex_metric("degree", DEGREE_PATTERN, span=True)
CurrencyRecall, CurrencyPrecision, CurrencyFscore = register_regex_metric("currency", CURRENCY_PATTERN, span=True)
MeasurementRecall, MeasurementPrecision, MeasurementFscore = register_regex_metric(
    "measurement", MEASUREMENT_PATTERN, span=True
)

__all__ = [
    "NumberRecall",
    "NumberPrecision",
    "NumberFscore",
    "PercentageRecall",
    "PercentagePrecision",
    "PercentageFscore",
    "DegreeRecall",
    "DegreePrecision",
    "DegreeFscore",
    "CurrencyRecall",
    "CurrencyPrecision",
    "CurrencyFscore",
    "MeasurementRecall",
    "MeasurementPrecision",
    "MeasurementFscore",
]
