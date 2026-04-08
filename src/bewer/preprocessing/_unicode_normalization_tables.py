"""Unicode variant → ASCII canonical form translation tables for character normalization.

For the split token tables (apostrophe, hyphen, slash), only punctuation-category
(Pd, Po, Sk, Lm) variants are included. Math-symbol category (Sm) characters such as
U+2212 MINUS SIGN (−) and U+2215 DIVISION SLASH (∕) are deliberately excluded: the
default tokenizer extracts Sm characters as separate tokens, after which the
normalizer's transliterate_symbols step already converts them to their ASCII
equivalents. Including them here would be redundant.
"""

APOSTROPHE_VARIANTS: dict[str, str] = {
    "\u0060": "'",  # GRAVE ACCENT
    "\u00b4": "'",  # ACUTE ACCENT
    "\u02bc": "'",  # MODIFIER LETTER APOSTROPHE
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201b": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2032": "'",  # PRIME
    "\uff07": "'",  # FULLWIDTH APOSTROPHE
}

HYPHEN_VARIANTS: dict[str, str] = {
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2012": "-",  # FIGURE DASH
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2015": "-",  # HORIZONTAL BAR
    "\u2e1a": "-",  # HYPHEN WITH DIAERESIS
    "\u2e3a": "-",  # TWO-EM DASH
    "\u2e3b": "-",  # THREE-EM DASH
    "\ufe31": "-",  # PRESENTATION FORM FOR VERTICAL EM DASH
    "\ufe32": "-",  # PRESENTATION FORM FOR VERTICAL EN DASH
    "\ufe58": "-",  # SMALL EM DASH
    "\ufe63": "-",  # SMALL HYPHEN-MINUS
    "\uff0d": "-",  # FULLWIDTH HYPHEN-MINUS
}

SLASH_VARIANTS: dict[str, str] = {
    "\uff0f": "/",  # FULLWIDTH SOLIDUS
}

# Pre-built str.maketrans tables
APOSTROPHE_TABLE = str.maketrans(APOSTROPHE_VARIANTS)
HYPHEN_TABLE = str.maketrans(HYPHEN_VARIANTS)
SLASH_TABLE = str.maketrans(SLASH_VARIANTS)
