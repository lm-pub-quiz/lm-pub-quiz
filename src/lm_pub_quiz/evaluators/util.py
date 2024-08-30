from typing import Dict, Generator, List, Optional, TypeVar

import torch
from transformers import BatchEncoding
from typing_extensions import reveal_type


class BaseMixin:
    def __init__(self, *args, **kwargs):
        pass


DataCollection = TypeVar("DataCollection", Dict, List, torch.Tensor, BatchEncoding)


def iter_batches(data_collection: DataCollection, batch_size: int) -> Generator[DataCollection, None, None]:
    """Yield successive n-sized chunks from tensors in provided dictionary."""

    if isinstance(data_collection, BatchEncoding):
        data = data_collection.data
    else:
        data = data_collection

    if isinstance(data, dict):
        if len(data) == 0:
            return
        else:
            # Determine the total number of elements
            n: Optional[int] = None
            for v in data.values():
                if n is not None and len(v) != n:
                    msg = "Values in batch have uneven sizes."
                    raise ValueError(msg)
                else:
                    n = len(v)
    else:
        n = len(data)

    assert n is not None

    if isinstance(data_collection, BatchEncoding):
        for i in range(0, n, batch_size):
            yield BatchEncoding({k: v[i : i + batch_size] for k, v in data.items()})

    elif isinstance(data_collection, dict):
        reveal_type(data_collection)
        for i in range(0, n, batch_size):
            yield {k: v[i : i + batch_size] for k, v in data.items()}

    else:
        for i in range(0, n, batch_size):
            yield data[i : i + batch_size]
