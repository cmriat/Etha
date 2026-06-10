"""Cell-to-slice mapping."""

import torch


def cell_slice(shape: torch.Size, num_slicers: list[int], cell: tuple[int, ...]) -> tuple[slice, ...]:
    ns = (num_slicers + [1] * len(shape))[: len(shape)]
    cell = (*cell, *(0,) * (len(shape) - len(cell)))
    return tuple(slice(c * (s // n), (c + 1) * (s // n)) for s, n, c in zip(shape, ns, cell, strict=True))
