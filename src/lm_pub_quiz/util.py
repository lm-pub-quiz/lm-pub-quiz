import itertools
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Union, cast, overload

from torch import Tensor
from transformers import BatchEncoding

from lm_pub_quiz.types import ItemScores, ItemTokenScoresAndRoles, ScoredToken, T, TokenRoles

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


def iter_batches(data_collection: T, batch_size: int) -> Iterable[T]:
    """Yield successive n-sized chunks from tensors in provided dictionary."""
    assert data_collection is not None

    if batch_size < 1:
        msg = "`batch_size` must be at least 1."
        raise ValueError(msg)

    if isinstance(data_collection, BatchEncoding):
        for batch, encodings in zip(
            iter_batches(data_collection.data, batch_size=batch_size),
            iter_batches(
                data_collection.encodings,
                batch_size=batch_size,
            )
            if data_collection.encodings is not None
            else itertools.repeat(None),
        ):
            yield BatchEncoding(batch, encoding=encodings)

    elif isinstance(data_collection, Mapping):
        keys, values = data_collection.keys(), data_collection.values()

        for batch_values in itertools.zip_longest(*(iter_batches(v, batch_size=batch_size) for v in values)):
            assert len(keys) == len(batch_values)
            yield dict(zip(keys, batch_values))

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
        seq: Sequence[T] = list(itertools.islice(it, current_size))

        assert len(seq) == current_size, "Iterable consumed before original shape could be reconstructed."

        yield seq


def tee_unzip(iterable: Iterable[tuple], n: int = 2) -> tuple[Iterable, ...]:
    iterators = itertools.tee(iterable, n)

    def it_select(it, i):
        return (x[i] for x in it)

    return tuple(it_select(iterators[i], i) for i in range(n))


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
