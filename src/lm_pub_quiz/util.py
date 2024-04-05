import os
from typing import Dict, List, Tuple, Union

PathLike = Union[str, os.PathLike]
ReducedReturnFormat = List[float]
EachTokenReturnFormat = List[Tuple[List[Tuple[str, float]], Dict[str, List[int]]]]
SegmentedResultFormat = Tuple[
    Tuple[List[str], ...], Tuple[List[float], ...], Tuple[List[int], ...], Tuple[List[int], ...], Tuple[List[int], ...]
]


def sort_scores(scores: List[float]) -> List[Tuple[int, float]]:
    """Sort (psudo) log likelihood scores (descending)."""
    indexed_list = list(enumerate(scores))
    indexed_list.sort(key=lambda x: x[1], reverse=True)
    return indexed_list


def parse_dumped_raw_results(result: EachTokenReturnFormat) -> SegmentedResultFormat:
    tokens, pll_scores = zip(*[([t for t, _ in sent], [p for _, p in sent]) for sent, _ in result])
    sub_indices, obj_indices, template_indices = zip(*((d["subject"], d["answer"], d["template"]) for _, d in result))
    return tokens, pll_scores, sub_indices, obj_indices, template_indices
