import inspect
from collections import namedtuple
from importlib import import_module

from omegaconf import OmegaConf

from bewer.core.flags import NORMALIZERS, STANDARDIZERS, TOKENIZERS
from bewer.preprocessing.normalization import Normalizer
from bewer.preprocessing.tokenization import Tokenizer

Pipelines = namedtuple("Pipelines", [STANDARDIZERS, TOKENIZERS, NORMALIZERS])


def resolve_function(path: str):
    """Resolve a function from a dot-separated path string."""
    module_name, func_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, func_name)


def resolve_func_pipeline(name, cfg):
    """Resolve a normalization pipeline from a configuration dictionary."""
    pipeline = []

    for func, cfg_params in cfg.items():
        cfg_params = cfg_params or {}
        norm_func = resolve_function(func)
        func_params = inspect.signature(norm_func).parameters.items()
        param_iter = iter(func_params)

        # Check that first positional argument is not passed in the config.
        first_param = next(param_iter)[0]
        if first_param in cfg_params:
            raise ValueError(f"First positional argument '{first_param}' should not be passed in params")

        # Check that all required parameters have a default value or are provided in the config.
        for param, value in param_iter:
            if value.default is inspect.Parameter.empty and param not in cfg_params:
                raise ValueError(f"Parameter '{param}' not found in function '{func}'")

        # Check for unexpected parameters in the config.
        for param in cfg_params:
            if param not in func_params:
                raise ValueError(f"Unexpected parameter '{param}' for function '{func}'")

        pipeline.append((norm_func, cfg_params))

    return Normalizer(pipeline, name)


def resolve_tokenizer(name, cfg):
    """Resolve a tokenizer from a configuration dictionary."""
    if len(cfg) != 1:
        raise ValueError("Tokenizer config must contain exactly one tokenizer definition")

    func, cfg_params = next(iter(cfg.items()))
    cfg_params = cfg_params or {}
    tokenizer_func = resolve_function(func)
    pattern = tokenizer_func(**cfg_params)
    return Tokenizer(pattern, name)


def resolve_pipelines(cfg: OmegaConf) -> dict[str, dict[str, callable]]:
    """Resolve normalization and tokenization pipelines from a configuration object."""

    pipelines = Pipelines(standardizers={}, tokenizers={}, normalizers={})

    for name, pipeline_cfg in cfg.standardizers.items():
        pipelines.standardizers[name] = resolve_func_pipeline(name, pipeline_cfg)
    for name, pipeline_cfg in cfg.tokenizers.items():
        pipelines.tokenizers[name] = resolve_tokenizer(name, pipeline_cfg)
    for name, pipeline_cfg in cfg.normalizers.items():
        pipelines.normalizers[name] = resolve_func_pipeline(name, pipeline_cfg)

    return pipelines
