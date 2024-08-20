import os
from pathlib import Path
from typing import List, Tuple

from lm_pub_quiz.types import EachTokenReturnFormat, SegmentedResultFormat

cache_base_path = Path(os.getenv("LM_PUB_QUIZ_CACHE_ROOT", Path(Path.home(), ".lm-pub-quiz")))


def sort_scores(scores: List[float]) -> List[Tuple[int, float]]:
    """Sort (psudo) log likelihood scores (descending)."""
    indexed_list = list(enumerate(scores))
    indexed_list.sort(key=lambda x: x[1], reverse=True)
    return indexed_list


def parse_dumped_raw_results(result: EachTokenReturnFormat) -> SegmentedResultFormat:
    tokens, pll_scores = zip(*[([t for t, _ in sent], [p for _, p in sent]) for sent, _ in result])
    sub_indices, obj_indices, template_indices = zip(*((d["subject"], d["answer"], d["template"]) for _, d in result))
    return tokens, pll_scores, sub_indices, obj_indices, template_indices
