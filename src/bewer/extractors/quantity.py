r"""Quantity patterns.

Regular-expression building blocks for *quantities* — numbers, optionally decorated with an
internationally-shared unit or symbol — grouped into categories so each can be scored
separately:

================  ==========================  ==========================
Category          Pattern                     Matches
================  ==========================  ==========================
``number``        :data:`NUMBER_PATTERN`      every number (incl. the ``5`` in ``5 mg``)
``percentage``    :data:`PERCENTAGE_PATTERN`  ``95%``, ``12 %``
``degree``        :data:`DEGREE_PATTERN`      ``37°C``, ``451 °F``
``currency``      :data:`CURRENCY_PATTERN`    ``$100``, ``100 €``
``measurement``   :data:`MEASUREMENT_PATTERN` ``5 mg``, ``120 mmHg``
================  ==========================  ==========================

The design is deliberately **language-agnostic**: the patterns key only on Western-Arabic
digits and a fixed set of internationally-shared symbols and SI/metric unit abbreviations,
never on words. So they apply uniformly across Latin-script languages without per-language
configuration — but they deliberately do **not** cover word-shaped quantities (spelled-out
numbers, ordinal suffixes, written month/unit names), which are inherently language-specific.

The categories are **not** mutually exclusive: ``number`` matches every number, including the
numeric part of a percentage, currency amount or measurement. Keeping them disjoint would mean
each pattern having to know about all the others, which is brittle, so categories may overlap.

The decorated categories are matched across token boundaries (the unit/symbol usually
tokenizes off the number), so they use ``span=True``; ``number`` full-matches whole number
tokens (``span=False``). :data:`QUANTITY_CATEGORIES` maps each name to its ``(pattern, span)``.
"""

from __future__ import annotations

import regex

__all__ = [
    "NUMBER",
    "PERCENT",
    "DEGREE",
    "CURRENCY",
    "UNIT",
    "NUMBER_PATTERN",
    "PERCENTAGE_PATTERN",
    "DEGREE_PATTERN",
    "CURRENCY_PATTERN",
    "MEASUREMENT_PATTERN",
    "QUANTITY_CATEGORIES",
]

# ---- components ----------------------------------------------------------------------

#: A number: digit runs joined by internal ``.``, ``,`` or ``:`` separators. Locale-neutral —
#: it accepts both ``3.5`` and ``3,5`` (and ``1,000`` / ``1.000``, ``14:30``) without trying
#: to interpret which separator means what, since extraction only needs to locate the term.
NUMBER = r"\p{Nd}+(?:[.,:]\p{Nd}+)*"

#: Percent and per-mille signs. (Note: the active ``orthographically_complex_term`` tokenizer does not retain
#: ``‰`` as its own token, so a per-mille value currently matches on its number alone; the
#: sign is included here so the pattern is correct if a symbol-retaining tokenizer is used.)
PERCENT = r"[%‰]"

#: A degree sign, optionally with a Celsius/Fahrenheit/Kelvin letter (``°``, ``°C``, ``°F``).
DEGREE = r"°[CFK]?"

#: Common currency signs (may lead or trail the number).
CURRENCY = r"[$€£¥₩₹¢]"

# SI and common metric/clinical unit symbols, longest first so the alternation prefers the
# most specific match. Case-sensitive on purpose: in SI, case is meaningful (mg vs Mg, mHz vs
# MHz), and the metrics run under a case-preserving normalizer.
_UNIT_SYMBOLS = [
    "mmHg",
    "kPa",
    "hPa",
    "mmol",
    "µmol",
    "kcal",
    "kHz",
    "MHz",
    "GHz",
    "mol",
    "min",
    "km",
    "cm",
    "mm",
    "µm",
    "nm",
    "kg",
    "mg",
    "µg",
    "ng",
    "dl",
    "cl",
    "ml",
    "ms",
    "µs",
    "ns",
    "kJ",
    "kW",
    "mV",
    "mA",
    "Hz",
    "Pa",
    "m",
    "g",
    "l",
    "L",
    "s",
    "h",
    "J",
    "W",
    "V",
    "A",
    "K",
    "N",
]
#: Alternation of recognised unit symbols, guarded so it does not match a unit-shaped prefix
#: of a longer word (e.g. the ``m`` in ``mph``).
UNIT = r"(?:" + "|".join(regex.escape(u) for u in _UNIT_SYMBOLS) + r")(?![\p{L}\p{Nd}])"

# ---- per-category patterns -----------------------------------------------------------

#: Every number (full token). Matches all numbers, including those that are part of a
#: percentage, currency amount or measurement (whose number tokenizes apart from the symbol).
NUMBER_PATTERN = NUMBER

#: A number followed by a percent sign.
PERCENTAGE_PATTERN = rf"{NUMBER}\s?{PERCENT}"

#: A number followed by a degree (optionally Celsius/Fahrenheit/Kelvin).
DEGREE_PATTERN = rf"{NUMBER}\s?{DEGREE}"

#: A currency amount: a currency sign leading or trailing a number.
CURRENCY_PATTERN = rf"(?:{CURRENCY}\s?{NUMBER}|{NUMBER}\s?{CURRENCY})"

#: A number followed by an SI / metric unit symbol.
MEASUREMENT_PATTERN = rf"{NUMBER}\s?{UNIT}"

#: Mapping of category name to ``(pattern, span)``. ``number`` is matched per token; the
#: decorated categories straddle token boundaries and so are span-matched.
QUANTITY_CATEGORIES: dict[str, tuple[str, bool]] = {
    "number": (NUMBER_PATTERN, False),
    "percentage": (PERCENTAGE_PATTERN, True),
    "degree": (DEGREE_PATTERN, True),
    "currency": (CURRENCY_PATTERN, True),
    "measurement": (MEASUREMENT_PATTERN, True),
}
