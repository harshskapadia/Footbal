"""Pick the compute device: GPU when available, CPU otherwise."""

from __future__ import annotations


def select_device(preference: str = "auto") -> str:
    """Resolve a device string for Ultralytics / PyTorch.

    Args:
        preference: ``"auto"`` picks CUDA, then Apple MPS, then CPU.
            Any other value (``"cpu"``, ``"cuda:0"``, ...) is returned unchanged.

    Returns:
        A device string such as ``"cuda:0"``, ``"mps"`` or ``"cpu"``.
    """
    if preference and preference != "auto":
        return preference
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
