"""MultipleChoiceDataset"""

import json
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing_extensions import Self

from lm_pub_quiz.data.base import DatasetBase, InstanceTableFileFormat, Item, RelationBase
from lm_pub_quiz.data.util import download_tmp_file, extract_archive_member, natural_sort
from lm_pub_quiz.types import PathLike
from lm_pub_quiz.util import cache_base_path

tqdm.pandas()


log = logging.getLogger(__name__)


KNOWN_DATASET_URLS: dict[str, tuple[str, Union[str, Callable]]] = {
    "bear": (
        "https://github.com/lm-pub-quiz/BEAR/archive/725b4e3139d0a5fdf914b0419ba744273dddc689.zip",
        "BEAR-725b4e3139d0a5fdf914b0419ba744273dddc689/BEAR",
    ),
    "bear-big": (
        "https://github.com/lm-pub-quiz/BEAR/archive/725b4e3139d0a5fdf914b0419ba744273dddc689.zip",
        "BEAR-725b4e3139d0a5fdf914b0419ba744273dddc689/BEAR-big",
    ),
}


class Relation(RelationBase):
    """
    Represents a relation within a dataset, including its code, answer space, templates, and an instance table.
    """

    def __init__(
        self,
        relation_code: str,
        *,
        templates: list[str],
        answer_space: Optional[pd.Series],
        instance_table: Optional[pd.DataFrame],
        lazy_options: Optional[dict[str, Any]],
        relation_info: Optional[dict[str, Any]] = None,
    ):
        if instance_table is None and lazy_options is None:
            msg = "Either instance_table of lazy_options must be specified"
            raise ValueError(msg)

        super().__init__(
            relation_code,
            instance_table=instance_table,
            answer_space=answer_space,
            lazy_options=lazy_options,
            relation_info=relation_info,
        )
        self.templates = templates

    def get_metadata(self, key: Optional[str] = None) -> Any:
        if key == "template":
            return self.templates
        elif key is not None:
            return super().get_metadata(key)
        else:
            return {
                "templates": self.templates,
                **super().get_metadata(),
            }

    @property
    def has_instance_table(self) -> bool:
        return True

    def subsample(self, n: int = 10) -> pd.DataFrame:
        """
        Returns only a subsampled version of the dataset of the size n.

        Parameters:
            n (int): Size of the subsampled dataset

        Returns:
            pd.DataFrame: Subsampled version of the dataset.
        """
        return self.instance_table.sample(n=n, random_state=23)

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        relation_code: Optional[str] = None,
        lazy: bool = True,
        fmt: InstanceTableFileFormat = None,
    ) -> Self:
        """
        Loads a relation from a JSONL file and associated metadata.

        Parameters:
            path (PathLike): The path to the dataset directory.
            relation_code (str): The specific code of the relation to load.
            lazy (bool): If False, the instance table is loaded directly into memory.

        Returns:
            Relation: An instance of the Relation class populated with data from the file.

        Raises:
            Exception: If there is an error in loading the file or processing the data.
        """

        path = Path(path)

        if not path.exists():
            msg = f"The provided path {path} does not exist."
            raise FileNotFoundError(msg)

        elif path.is_file():
            relation_path = path
            dataset_path = relation_path.parent

            if relation_code is None:
                relation_code = cls.code_from_path(relation_path)

        elif relation_code is not None:
            dataset_path = path
            relation_path = cls.search_path(dataset_path, relation_code=relation_code, fmt=fmt)

            if relation_path is None:
                if fmt is None:
                    fmt_info = ""
                else:
                    fmt_info = f" and format {fmt}"
                msg = f"No file with the relation code {relation_code}{fmt_info} could be found in path {dataset_path}."
                raise FileNotFoundError(msg)

        else:
            # directory passed, but no relation code given
            msg = "A path to a directory was passed but no relation was specified."
            raise ValueError(msg)

        assert relation_code is not None

        log.debug("Loading %s (%s) from: %s", cls.__name__, relation_code, relation_path)

        metadata_path = dataset_path / cls._metadata_file_name

        with open(metadata_path) as meta_file:
            try:
                metadata = json.load(meta_file)[relation_code]
            except KeyError as e:
                msg = f"Relation '{relation_code}' (from file '{path}') not found in '{metadata_path}'."
                raise KeyError(msg) from e
            templates = metadata.pop("templates", [])

        answer_space = cls.answer_space_from_metadata(metadata, id_prefix=f"{relation_code}-")

        if lazy:
            instance_table = None
            lazy_options = {
                "path": relation_path,
                "fmt": fmt,
            }
        else:
            instance_table = cls.load_instance_table(relation_path, answer_space=answer_space)
            lazy_options = None

        return cls(
            relation_code,
            answer_space=answer_space,
            templates=templates,
            instance_table=instance_table,
            lazy_options=lazy_options,
            relation_info=metadata,
        )

    def copy(self, *, relation_code: Optional[str] = None, **kw):
        kw = {
            "templates": self.templates.copy(),
            **kw,
        }
        return super().copy(relation_code=relation_code, **kw)

    def filter_subset(
        self,
        indices: Sequence[int],
        *,
        keep_answer_space: bool = False,
    ) -> Self:
        original_instance_table = self.instance_table
        instance_table = original_instance_table.iloc[indices].copy().reset_index()

        if keep_answer_space and self._answer_space is not None:
            answer_space = self._answer_space
        elif keep_answer_space:
            answer_space = self.answer_space_from_instance_table(original_instance_table)
        else:
            answer_space = self.answer_space_from_instance_table(instance_table)

            answer_space_translation = {
                self.answer_space.index.get_loc(_id): i for i, _id in enumerate(answer_space.index)
            }

            instance_table.answer_idx = instance_table["answer_idx"].map(answer_space_translation)

        return self.copy(
            answer_space=answer_space,
            instance_table=instance_table,
        )

    @classmethod
    def load_instance_table(
        cls, path: Path, *, answer_space: Optional[pd.Series] = None, fmt: InstanceTableFileFormat = None
    ) -> pd.DataFrame:
        instance_table = super().load_instance_table(path, answer_space=answer_space, fmt=fmt)

        if "obj_id" in instance_table and "answer_idx" not in instance_table:
            if answer_space is None:
                answer_space = cls.answer_space_from_instance_table(instance_table)
            answer_indices = pd.Series(np.arange(len(answer_space)), index=answer_space.index, name="answer_idx")
            instance_table = instance_table.join(answer_indices, on="obj_id")
        return instance_table

    def get_items(
        self,
        *,
        template_index: Union[int, Sequence[int], None] = None,
        subsample: Optional[int] = None,
        use_tqdm: bool = True,
    ) -> Iterator[Item]:
        df = self.instance_table if subsample is None else self.subsample(subsample)

        if template_index is None:
            template_index = list(range(len(self.templates)))
        elif isinstance(template_index, int):
            template_index = [template_index]

        instances = (
            Item.from_kw(
                subject=(row_dict := row.to_dict()).pop("sub_label"),
                answers=self.answer_space.tolist(),
                template=self.templates[t_index],
                template_index=t_index,
                instance_index=instance_index,
                answer_idx=row_dict.pop("answer_idx"),
                **row_dict,
            )
            for instance_index, row in df.iterrows()
            for t_index in template_index
        )

        if use_tqdm:
            return tqdm(instances, total=len(df) * len(template_index), desc=f"Relation {self.relation_code}")
        else:
            return instances


class Dataset(DatasetBase[Relation]):
    """
    A collection of relations forming a multiple choice dataset.

    Usage:
        The prefferred way to load the BEAR knowledge probe is to load it by name:

        >>> from lm_pub_quiz import Dataset
        >>> dataset = Dataset.from_name("BEAR")
    """

    def __init__(self, relations: list[Relation], path: PathLike, name: Optional[str] = None):
        self.relation_data = relations
        self.path = path
        self.name = name

    def __str__(self) -> str:
        if self.name is not None:
            relations_repr = ", ".join(self.relation_codes)
            return f"{self.__class__.__name__}({self.name}: {relations_repr})"
        else:
            return super().__str__()

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        lazy: bool = True,
        fmt: InstanceTableFileFormat = None,
        relation_info: Optional[PathLike] = None,
        **kwargs,
    ) -> Self:
        """
        Loads a multiple choice dataset from a specified directory path.

        This method scans the directory for relation files and assembles them into a MultipleChoiceDataset.

        Parameters:
            path (str): The directory path where the dataset is stored.
            lazy (bool): If False, the instance tables of all relations are directly loaded into memory.

        Returns:
            Dataset: An instance if Dataset loaded with the relations from the directory.

        Raises:
            Exception: If there is an error in loading the dataset.

        Usage:
            Loading the BEAR-dataset.
            ``` python
            >>> from lm_pub_quiz import Dataset
            >>> dataset = Dataset.from_path("/path/to/dataset/BEAR")
            ```
        """
        kwargs["path"] = path
        dataset_path = Path(path)

        if not dataset_path.exists():
            msg = f"Dataset at `{dataset_path}` could not be opened: Path does not exist."
            raise RuntimeError(msg)

        relation_files = natural_sort(Relation.search_path(dataset_path, fmt=fmt))
        relations = [Relation.from_path(p, lazy=lazy) for p in relation_files]

        # if no name was passed, default to using the name of the dataset directory
        if "name" not in kwargs:
            kwargs["name"] = dataset_path.stem

        log.info("Loaded dataset `%s` (%d relations) from `%s`.", kwargs["name"], len(relation_files), dataset_path)

        obj = cls(relations, **kwargs)

        if relation_info is not None:
            with open(relation_info) as f:
                obj.update_relation_info(json.load(f))

        return obj

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        lazy: bool = True,
        base_path: Optional[Path] = None,
        chunk_size: int = 10 * 1024,
        relation_info: Optional[PathLike] = None,
        **kwargs,
    ) -> Self:
        """
        Loads a dataset from the cache (if available) or the url which is specified in the internal dataset table.

        Parameters:
            name (str): The name of the dataset.
            lazy (bool): If False, the instance tables of all relations are directly loaded into memory.

        Returns:
            Dataset: An instance if Dataset loaded with the relations from the directory.

        Raises:
            Exception: If there is an error in loading the dataset.

        Usage:
            Loading the BEAR-dataset.
            ``` python
            >>> from lm_pub_quiz import Dataset
            >>> dataset = Dataset.from_name("BEAR")
            ```
        """
        # Check if dataset exists in cache
        dataset_path = (base_path if base_path is not None else cache_base_path / "datasets") / name.lower()

        if not dataset_path.exists():
            # Check wether the dataset is known (in internal dataset-url table)
            try:
                url, extraction_info = KNOWN_DATASET_URLS[name.lower()]
            except KeyError as e:
                msg = f"Could not find dataset '{name}' in cache (or a matching dataset in the url table)."
                raise KeyError(msg) from e

            log.info("%s not found in chache, downloading from provided url: %s", name, url)

            _, path = download_tmp_file(url, desc=name, chunk_size=chunk_size)

            dataset_path.mkdir(parents=True, exist_ok=True)

            if isinstance(extraction_info, str):
                # Assume we want to extract the member of a zip archive
                extract_archive_member(
                    source=path,
                    target=dataset_path,
                    member=extraction_info,
                )

            else:
                # Assume a function for extraction is given
                extraction_info(
                    source=path,
                    target=dataset_path,
                )

            # Clean up
            os.remove(path)
        else:
            log.debug("Dataset %s found in cache at %s.", name, dataset_path)

        return cls.from_path(dataset_path, lazy=lazy, name=name, relation_info=relation_info, **kwargs)

    def activated(self):
        if not self.is_lazy:
            return self

        return self.__class__(name=self.name, path=self.path, relations=[rel.activated() for rel in self])

    def filter_subset(
        self,
        indices: Mapping[str, Sequence[int]],
        *,
        save_path: Optional[PathLike] = None,
        fmt: InstanceTableFileFormat = None,
        dataset_name: Optional[str] = None,
        keep_answer_space: bool = False,
    ) -> Self:
        relations: list[Relation] = []

        for key, value in indices.items():
            # filter the relation
            rel = self[key].filter_subset(value, keep_answer_space=keep_answer_space)

            # if save_path is fiven, save and replace with lazy-loading relation
            if save_path is not None:
                rel = rel.saved(path=save_path, fmt=fmt)

            relations.append(rel)

        return self.__class__(
            relations,
            name=dataset_name if dataset_name is not None else f"{self.name} (subset)",
            path=save_path if save_path is not None else Path("."),
        )
