"""Progress reporting that works for both the CLI (tqdm) and the Streamlit app."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")
ProgressCallback = Callable[[float, str], None]


def progress(
    items: Iterable[T], total: int | None, desc: str, callback: ProgressCallback | None = None
) -> Iterator[T]:
    """Yield ``items`` while showing a tqdm bar and calling ``callback(fraction, desc)``."""
    step = max(1, (total or 100) // 50)
    for i, item in enumerate(tqdm(items, total=total, desc=desc)):
        if callback is not None and total and i % step == 0:
            callback(min(i / total, 1.0), desc)
        yield item
    if callback is not None:
        callback(1.0, desc)
