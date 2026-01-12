from contextlib import contextmanager
from contextvars import ContextVar

from bewer.flags import DEFAULT

# Define the context variables with default value "default"
STANDARDIZER_NAME = ContextVar("STANDARDIZER", default=DEFAULT)
TOKENIZER_NAME = ContextVar("TOKENIZER", default=DEFAULT)
NORMALIZER_NAME = ContextVar("NORMALIZER", default=DEFAULT)


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
