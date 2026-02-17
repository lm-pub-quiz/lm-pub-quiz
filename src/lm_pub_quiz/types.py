import os
from collections.abc import Sequence
from typing import Callable, TypeVar, Union

from transformers import CharSpan

PathLike = Union[str, os.PathLike]

ScoringMask = list[bool]  # Denotes tokens which should be used to score a sequence

TextSelection = list[CharSpan]  # Marks a (potentially discontinuous section) of the text
TextRoles = dict[str, TextSelection]  # Maps a role (template, subject, object) to a list of spans

TokenRoles = dict[str, list[int]]  # Maps a role (template, subject, object) to the respective tokens


StatementScore = float
ScoredToken = tuple[str, float]  # (token string, token PLL score)
TokenScoresAndRoles = tuple[list[ScoredToken], TokenRoles]

ItemScores = Sequence[StatementScore]
ItemTokenScoresAndRoles = Sequence[TokenScoresAndRoles]

SegmentedResultFormat = tuple[
    tuple[list[str], ...],  # Token strings
    tuple[list[float], ...],  # PLL scores for these tokens
    tuple[list[int], ...],  # tokens that are part of the subject
    tuple[list[int], ...],  # tokens that are part of the object
    tuple[list[int], ...],  # Tokens that are part of the template
]

T = TypeVar("T")
V = TypeVar("V")

ReductionFunction = Callable[..., StatementScore]
