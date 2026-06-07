"""Library of pre-defined vocabulary extractor functions.

Each extractor here is a :data:`~bewer.core.vocabulary.VocabularyExtractor` — a callable
``(dataset) -> terms`` that can be registered with a dataset via
:meth:`Dataset.add_vocabulary_from_function` (or wrapped in a :class:`Vocabulary` via
:meth:`Vocabulary.from_function`) to derive key terms on demand.
"""

from bewer.extractors.complex_term import COMPLEX_TERM_DEFAULT_PATTERN, ComplexTermExtractor

__all__ = ["COMPLEX_TERM_DEFAULT_PATTERN", "ComplexTermExtractor"]
