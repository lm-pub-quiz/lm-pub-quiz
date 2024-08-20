import os
from typing import Dict, List, Tuple, Union

PathLike = Union[str, os.PathLike]

ScoredToken = Tuple[str, float]  # (token string, token PLL score)

ReducedReturnFormat = List[float]

ScoringMask = List[bool]  # Denotes tokens which should be used to score a sequence

Span = Tuple[int, int]  # Marks a section of the string
SpanRoles = Dict[str, List[Span]]  # Maps a role (template, subject, object) to a list of spans

TokenRoles = Dict[str, List[int]]  # Maps a role (template, subject, object) to the respective tokens

EachTokenReturnFormat = List[Tuple[List[ScoredToken], TokenRoles]]  # List of (token string, token PLL score)

SegmentedResultFormat = Tuple[
    Tuple[List[str], ...],  # Token strings
    Tuple[List[float], ...],  # PLL scores for these tokens
    Tuple[List[int], ...],  # tokens that are part of the subject
    Tuple[List[int], ...],  # tokens that are part of the object
    Tuple[List[int], ...],  # Tokens that are part of the template
]
