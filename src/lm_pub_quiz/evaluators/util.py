from typing import Dict, Iterable, Optional, TypeVar, Union, overload

import torch

T = TypeVar("T", list, torch.Tensor)


class BaseMixin:
    def __init__(self, *args, **kwargs):
        pass


@overload
def iter_batches(batch: T, batch_size: int) -> Iterable[T]: ...


@overload
def iter_batches(batch: Dict[str, T], batch_size: int) -> Iterable[Dict[str, T]]: ...


def iter_batches(batch: Union[T, Dict[str, T]], batch_size: int) -> Union[Iterable[T], Iterable[Dict[str, T]]]:
    """Yield successive n-sized chunks from tensors in provided dictionary."""
    if isinstance(batch, dict):
        if len(batch) == 0:
            return
        else:
            # Determine the total number of elements
            n: Optional[int] = None
            for v in batch.values():
                if n is not None and len(v) != n:
                    msg = "Values in batch have uneven sizes."
                    raise ValueError(msg)
                else:
                    n = len(v)
    else:
        n = len(batch)

    assert n is not None

    for i in range(0, n, batch_size):
        if isinstance(batch, dict):
            yield {k: v[i : i + batch_size] for k, v in batch.items()}
        else:
            yield batch[i : i + batch_size]
