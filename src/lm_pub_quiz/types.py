import os
from typing import Union

PathLike = Union[str, os.PathLike]

ScoredToken = tuple[str, float]  # (token string, token PLL score)

ReducedReturnFormat = list[float]

ScoringMask = list[bool]  # Denotes tokens which should be used to score a sequence

Span = tuple[int, int]  # Marks a section of the string
SpanRoles = dict[str, list[Span]]  # Maps a role (template, subject, object) to a list of spans

TokenRoles = dict[str, list[int]]  # Maps a role (template, subject, object) to the respective tokens

EachTokenReturnFormat = list[tuple[list[ScoredToken], TokenRoles]]  # List of (token string, token PLL score)

SegmentedResultFormat = tuple[
    tuple[list[str], ...],  # Token strings
    tuple[list[float], ...],  # PLL scores for these tokens
    tuple[list[int], ...],  # tokens that are part of the subject
    tuple[list[int], ...],  # tokens that are part of the object
    tuple[list[int], ...],  # Tokens that are part of the template
]
