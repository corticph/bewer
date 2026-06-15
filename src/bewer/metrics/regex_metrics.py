"""Factory for *regex metrics*.

A regex metric is a recall / precision / F-score trio computed over a vocabulary that is
auto-extracted from the references by a regular expression. It generalises the key-term
metrics (KTR / KTP / KTF): instead of an enumerated key-term list, the terms are whatever a
pattern matches. The predefined metrics — ``orthographically_complex_term``, ``number``,
``percentage``, ``degree``, ``currency`` and ``measurement`` — are a flat set of such instances; none is
special relative to another, and a user-registered metric is their equal.

Writing three metric classes (plus param schemas and vocabulary auto-registration) by hand
for every such family is pure boilerplate. :func:`register_regex_metric` collapses it to one
call — this is the supported way to add your own regex metrics, no class definitions needed::

    register_regex_metric("measurement", MEASUREMENT_PATTERN, span=True)

    dataset.metrics.measurement_recall().value
    dataset.metrics.measurement_precision().value
    dataset.metrics.measurement_fscore(beta=1.0).value

It registers ``<base>_recall`` / ``<base>_precision`` / ``<base>_fscore`` (subclasses of
KTR / KTP / KTF) over a ``<base>_terms`` vocabulary backed by a
:class:`~bewer.extractors.regex.RegexExtractor`, auto-registered on the dataset on first use.
"""

from __future__ import annotations

from dataclasses import field, make_dataclass
from typing import Optional, Union

import regex

from bewer.core.vocabulary import Vocabulary
from bewer.extractors.regex import RegexExtractor
from bewer.metrics.base import METRIC_REGISTRY, Metric
from bewer.metrics.ktf import KTF
from bewer.metrics.ktp import KTP
from bewer.metrics.ktr import KTR

__all__ = ["register_regex_metric"]


def _make_auto_vocab_params(base_params: type, vocab: str, extractor_factory) -> type:
    """Build a param dataclass that defaults ``vocab`` to ``vocab`` and auto-registers it.

    Subclasses ``base_params`` (KTR/KTP/KTF's ``param_schema``); on validate it registers the
    extractor-backed vocabulary if absent, then runs the base validation. Built with
    ``make_dataclass`` because a class body cannot read these enclosing-scope values.
    """

    def validate(self) -> None:
        dataset = self.metric.dataset
        if self.vocab not in dataset._vocabularies:
            dataset._register_derived_vocabulary(Vocabulary(self.vocab).add_extractor(extractor_factory()))
        base_params.validate(self)

    return make_dataclass(
        f"{base_params.__qualname__.split('.')[0]}AutoParams",
        [("vocab", str, field(default=vocab))],
        bases=(base_params,),
        namespace={"validate": validate},
    )


def register_regex_metric(
    base: str,
    pattern: Union[str, "regex.Pattern"],
    *,
    span: bool = False,
    vocab: Optional[str] = None,
    tokenizer: str = "orthographically_complex_term",
    normalizer: str = "cased",
    label: Optional[str] = None,
    allow_override: bool = False,
) -> tuple[type[Metric], type[Metric], type[Metric]]:
    """Register a recall/precision/F-score metric set over a regex-extracted vocabulary.

    Args:
        base: Metric name stem. Registers ``<base>_recall``, ``<base>_precision`` and
            ``<base>_fscore``, accessible as ``dataset.metrics.<base>_recall()`` etc.
        pattern: The regular expression identifying the terms (string or compiled).
        span: Match strategy — ``True`` searches the pattern over the whole standardized text
            (for terms that straddle token boundaries, e.g. ``5 mg``); ``False`` full-matches
            it against each token. See :class:`~bewer.extractors.regex.RegexExtractor`.
        vocab: Vocabulary name. Defaults to ``f"{base}_terms"``. Auto-registered on first use.
        tokenizer: Tokenizer pipeline the metrics run under.
        normalizer: Normalizer pipeline the metrics run under.
        label: Human-readable label for metric names (e.g. "Measurement"). Defaults to ``base``
            title-cased with underscores replaced by spaces.
        allow_override: Passed through to the registry to permit re-registration.

    Returns:
        The ``(recall, precision, fscore)`` metric classes, in that order.
    """
    vocab = vocab or f"{base}_terms"
    label = label or base.replace("_", " ").title()

    def extractor_factory() -> RegexExtractor:
        return RegexExtractor(pattern, span=span)

    specs = [
        (f"{base}_recall", KTR, "Recall", "TP / (TP + FN) — the fraction transcribed correctly"),
        (f"{base}_precision", KTP, "Precision", "TP / (TP + FP) — how precisely the produced ones were transcribed"),
        (f"{base}_fscore", KTF, "F-Score", "the weighted harmonic mean of precision and recall"),
    ]

    registered: list[type[Metric]] = []
    for name, base_metric, kind, blurb in specs:
        params = _make_auto_vocab_params(base_metric.param_schema, vocab, extractor_factory)
        description = (
            f"{label} {kind.lower()}: key-term {kind.lower()} over the auto-extracted {label.lower()} "
            f"vocabulary (its terms are identified by a regular expression). {blurb}."
        )
        cls = type(
            "".join(part.capitalize() for part in name.split("_")),
            (base_metric,),
            {
                "short_name_base": name,
                "long_name_base": f"{label} {kind}",
                "description": description,
                "param_schema": params,
            },
        )
        METRIC_REGISTRY.register_metric(
            cls, name, tokenizer=tokenizer, normalizer=normalizer, allow_override=allow_override
        )
        registered.append(cls)

    return registered[0], registered[1], registered[2]
