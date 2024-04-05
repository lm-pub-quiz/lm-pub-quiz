import re
from pathlib import Path
from typing import Iterable, List, Tuple, Union


def _natural_sort_convert(text: str) -> Union[str, int]:
    return int(text) if text.isdigit() else text.lower()


_natural_sort_key_pattern = re.compile("([0-9]+)")


def _natural_sort_key(key: str) -> Tuple[Union[str, int], ...]:
    return tuple(_natural_sort_convert(c) for c in re.split(_natural_sort_key_pattern, key))


def natural_sort(iterable: Iterable[Union[str, Path]]) -> List[str]:
    """Source: https://stackoverflow.com/a/4836734"""
    return sorted(map(str, iterable), key=_natural_sort_key)
