"""Argus: PatchCore memory size vs calibrated false-positive rate."""

__version__ = "0.1.0"
__all__ = ["Detector", "load_detector"]


def __getattr__(name: str):
    # Lazy, so `import argus` (and scripts like argus.data.verify) don't need torch.
    if name in __all__:
        from argus import detector

        return getattr(detector, name)
    raise AttributeError(f"module 'argus' has no attribute {name!r}")
