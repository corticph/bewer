__all__ = [
    "display_basic_aligned",
    "ColorScheme",
    "DefaultColorScheme",
]


def __getattr__(name: str):
    if name in __all__:
        from bewer.reporting.python.alignment import (
            ColorScheme,
            DefaultColorScheme,
            display_basic_aligned,
        )

        _attrs = {
            "display_basic_aligned": display_basic_aligned,
            "ColorScheme": ColorScheme,
            "DefaultColorScheme": DefaultColorScheme,
        }
        return _attrs[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
