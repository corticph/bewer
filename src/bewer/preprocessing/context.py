from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

from bewer.flags import DEFAULT

if TYPE_CHECKING:
    from bewer import Dataset
    from bewer.preprocessing.normalization import Normalizer
    from bewer.preprocessing.tokenization import Tokenizer

# Define the context variables with default value "default"
STANDARDIZER_NAME = ContextVar("STANDARDIZER", default=DEFAULT)
TOKENIZER_NAME = ContextVar("TOKENIZER", default=DEFAULT)
NORMALIZER_NAME = ContextVar("NORMALIZER", default=DEFAULT)

# Ordered pipeline stages: (context_var, pipelines attribute name)
PIPELINE_STAGES = (
    (STANDARDIZER_NAME, "standardizers"),
    (TOKENIZER_NAME, "tokenizers"),
    (NORMALIZER_NAME, "normalizers"),
)


@contextmanager
def set_pipeline(standardizer=DEFAULT, tokenizer=DEFAULT, normalizer=DEFAULT):
    tok_token = TOKENIZER_NAME.set(tokenizer)
    std_token = STANDARDIZER_NAME.set(standardizer)
    norm_token = NORMALIZER_NAME.set(normalizer)
    try:
        yield
    finally:
        TOKENIZER_NAME.reset(tok_token)
        STANDARDIZER_NAME.reset(std_token)
        NORMALIZER_NAME.reset(norm_token)


def get_standardizer(dataset: "Dataset") -> Optional["Normalizer"]:
    """Get the standardizer for the dataset in the current context."""
    return dataset.pipelines.standardizers.get(STANDARDIZER_NAME.get(), None)


def get_tokenizer(dataset: "Dataset") -> Optional["Tokenizer"]:
    """Get the tokenizer for the dataset in the current context."""
    return dataset.pipelines.tokenizers.get(TOKENIZER_NAME.get(), None)


def get_normalizer(dataset: "Dataset") -> Optional["Normalizer"]:
    """Get the normalizer for the dataset in the current context."""
    return dataset.pipelines.normalizers.get(NORMALIZER_NAME.get(), None)
