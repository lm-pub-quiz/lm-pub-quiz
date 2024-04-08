import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple, Union
from zipfile import ZipFile

import requests
from tqdm import tqdm

from lm_pub_quiz.util import PathLike


def _natural_sort_convert(text: str) -> Union[str, int]:
    return int(text) if text.isdigit() else text.lower()


_natural_sort_key_pattern = re.compile("([0-9]+)")


def _natural_sort_key(key: str) -> Tuple[Union[str, int], ...]:
    return tuple(_natural_sort_convert(c) for c in re.split(_natural_sort_key_pattern, key))


def natural_sort(iterable: Iterable[Union[str, Path]]) -> List[str]:
    """Source: https://stackoverflow.com/a/4836734"""
    return sorted(map(str, iterable), key=_natural_sort_key)


def download_tmp_file(url, chunk_size: int = 10 * 1024, desc: str = "") -> Tuple[int, Path]:
    """Download file from `url` and save in a temporary directory."""
    with requests.get(
        url,
        headers={"User-Agent": "LM Pub Quiz"},
        allow_redirects=True,
        stream=True,
        timeout=10,
    ) as response:
        response.raise_for_status()

        fd, path = tempfile.mkstemp()

        with os.fdopen(fd, "wb") as tmp_file, tqdm(
            desc=desc,
            total=int(response.headers.get("content-length", 0)),
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:

            for chunk in response.iter_content(chunk_size=chunk_size):
                size = tmp_file.write(chunk)
                bar.update(size)

        return fd, Path(path)


def extract_archive_member(source: PathLike, target: PathLike, member: str):
    """
    Extract `member` from the archive at `source_path` and saves it as `target_path`.
    """
    with ZipFile(source) as archive:
        for info in archive.infolist():
            # Check whether to extract the member
            if not info.is_dir() and info.filename.startswith(member + "/"):
                # modify the name such that it is relative to the member argument
                new_name = info.filename[len(member) + 1 :]
                info.filename = new_name

                # extrat the member infot the specified path
                archive.extract(info, path=target)
