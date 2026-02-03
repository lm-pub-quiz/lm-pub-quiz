import itertools
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Union, cast, overload

from torch import Tensor
from transformers import BatchEncoding

from lm_pub_quiz.types import ItemScores, ItemTokenScoresAndRoles, ScoredToken, T, TokenRoles, V

cache_base_path = Path(os.getenv("LM_PUB_QUIZ_CACHE_ROOT", Path(Path.home(), ".lm-pub-quiz")))


def sort_scores(scores: list[float]) -> list[tuple[int, float]]:
    """Sort (psudo) log likelihood scores (descending)."""
    indexed_list = list(enumerate(scores))
    indexed_list.sort(key=lambda x: x[1], reverse=True)
    return indexed_list


def parse_dumped_raw_results(result: Union[ItemTokenScoresAndRoles, ItemScores]) -> dict[str, Any]:
    row = {}

    if any(isinstance(r, (int, float)) for r in result):
        result = cast(ItemScores, result)
        row["pll_scores"] = result
    else:
        result = cast(ItemTokenScoresAndRoles, result)
        row["tokens"], row["pll_scores"] = [], []
        row["sub_indices"], row["obj_indices"], row["template_indices"] = [], [], []

        token_scores: list[ScoredToken]
        roles: TokenRoles

        for token_scores, roles in result:
            tokens, scores = zip(*token_scores)
            row["tokens"].append(tokens)
            row["pll_scores"].append(scores)

            row["sub_indices"].append(roles["subject"])
            row["obj_indices"].append(roles["answer"])
            row["template_indices"].append(roles["template"])

        # row["tokens"], row["pll_scores"] = zip(*[([t for t, _ in sent], [p for _, p in sent]) for sent, _ in result])
        # row["sub_indices"], row["obj_indices"], row["template_indices"] = zip(
        #    *((d["subject"], d["answer"], d["template"]) for _, d in result)
        # )
    return row


@overload
def iter_batches(data_collection: Iterable[V], batch_size: int) -> Iterable[Sequence[V]]: ...


@overload
def iter_batches(data_collection: Tensor, batch_size: int) -> Iterable[Tensor]: ...


@overload
def iter_batches(data_collection: BatchEncoding, batch_size: int) -> Iterable[BatchEncoding]: ...


@overload
def iter_batches(data_collection: dict[T, Iterable[V]], batch_size: int) -> Iterable[dict[T, Sequence[V]]]: ...


def iter_batches(
    data_collection: Union[Iterable[V], Tensor, BatchEncoding, dict[T, Iterable[V]]], batch_size: int
) -> Union[Iterable[list[V]], Iterable[Tensor], Iterable[BatchEncoding], Iterable[dict[T, list[V]]]]:
    """Yield successive n-sized chunks from tensors in provided dictionary."""

    if batch_size < 1:
        msg = "`batch_size` must be at least 1."
        raise ValueError(msg)

    if isinstance(data_collection, BatchEncoding):
        for batch, encodings in zip(
            iter_batches(data_collection.data, batch_size=batch_size),
            iter_batches(data_collection.encodings, batch_size=batch_size),
        ):
            yield BatchEncoding(batch, encoding=encodings)

    elif isinstance(data_collection, dict):
        keys, values = data_collection.keys(), data_collection.values()

        for batches in zip(iter_batches(v, batch_size=batch_size) for v in values):
            yield dict(zip(keys, batches))

    elif isinstance(data_collection, Tensor):
        for i in range(0, data_collection.size(0), batch_size):
            yield data_collection[i : i + batch_size]

    else:
        it = iter(data_collection)

        while batch := tuple(itertools.islice(it, batch_size)):
            yield list(batch)


def chain_with_sizes(iterable: Iterable[Sequence[T]]) -> tuple[Iterator[int], Iterator[T]]:
    it_size, it_elements = itertools.tee(iterable, 2)

    return iter(map(len, it_size)), itertools.chain.from_iterable(it_elements)


def unchain(iterable: Iterable[T], *, sizes: Iterable[int]) -> Iterator[Sequence[T]]:
    it: Iterator[T] = iter(iterable)
    for current_size in sizes:
        yield [next(it) for _ in range(current_size)]


def tee_unzip(iterable: Iterable[tuple], n: int = 2) -> tuple[Iterable, ...]:
    iterators = itertools.tee(iterable, n)

    return tuple((t[i] for t in iterators[i]) for i in range(n))


class ReversibleChain(Mapping[str, Iterator]):
    def __init__(
        self,
        iterables: Mapping[str, Iterable],
    ):
        self._input_iterator: Iterator[dict[str, Sequence]]
        self._iterators: dict[str, Iterator] = {}
        self.sizes: Iterator[int]

        size_iterators: list[Iterable[int]] = []

        reverse_iterators: dict[str, Iterator[Sequence]] = {}
        forward_iterators: dict[str, Iterator[Sequence]] = {}

        for k, v in iterables.items():
            reverse_iterators[k], forward_iterators[k] = itertools.tee(v, 2)

        self._input_iterator = iter(
            dict(zip(reverse_iterators.keys(), sequences)) for sequences in zip(*reverse_iterators.values())
        )

        for k, v in forward_iterators.items():
            s, self._iterators[k] = chain_with_sizes(v)
            size_iterators.append(s)

        self.sizes = iter(map(self.ensure_matching_size, itertools.zip_longest(*size_iterators)))

    @staticmethod
    def ensure_matching_size(sizes):
        if any(s != sizes[0] for s in sizes):
            msg = f"Sizes are not of equal length: {', '.join(map(str, sizes))}"
            raise ValueError(msg)

        return sizes[0]

    @overload
    def reverse(self, iterable: Iterable[T], *, yield_inputs: Literal[False] = False) -> Iterator[Sequence[T]]: ...

    @overload
    def reverse(
        self, iterable: Iterable[T], *, yield_inputs: Literal[True]
    ) -> Iterator[tuple[Sequence[T], dict[str, Sequence]]]: ...

    def reverse(
        self, iterable: Iterable[T], *, yield_inputs: bool = False
    ) -> Union[Iterator[Sequence[T]], Iterator[tuple[Sequence[T], dict[str, Sequence]]]]:
        for inputs, result in zip(self._input_iterator, unchain(iterable, sizes=self.sizes)):
            if yield_inputs:
                yield inputs, result
            else:
                yield result

    def __getitem__(self, key: str) -> Iterator:
        return self._iterators[key]

    def __len__(self) -> int:
        return len(self._iterators)

    def __iter__(self) -> Iterator[str]:
        return iter(self._iterators)
