"""MultipleChoiceDataset"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Self

from lm_pub_quiz.data.base import DatasetBase, RelationBase
from lm_pub_quiz.data.util import natural_sort
from lm_pub_quiz.util import PathLike

log = logging.getLogger(__name__)


class Relation(RelationBase):
    """
    Represents a relation within a dataset, including its code, answer space, templates, and an instance table.

    Attributes:
        relation_code (str): A unique code identifying the relation.
        answer_space (List[str]): A list of possible answers for this relation.
        templates (List[str]): Templates for generating instances of this relation.
        instance_table (pd.DataFrame): A pandas DataFrame containing instances of the relation.

    Methods:
        __str__: Returns a string representation showing the first five instances in the relation.
        __repr__: Returns a string representation of the relation code.
        __len__: Returns the number of instances in the relation.
        subsample: Randomly samples a subset of instances from the relation.
        load_from_file: Class method to create a Relation instance from a JSONL file.
    """

    relation_code: str

    def __init__(
        self,
        relation_code: str,
        *,
        templates: List[str],
        answer_space: Optional[pd.Series],
        instance_table: Optional[pd.DataFrame],
        lazy_load_path: Optional[Path],
    ):
        if instance_table is None and lazy_load_path is None:
            msg = "Either instance_table of lazy_load_path must be specified"
            raise ValueError(msg)

        super().__init__(
            relation_code, instance_table=instance_table, answer_space=answer_space, lazy_load_path=lazy_load_path
        )
        self.templates = templates

    def get_metadata(self) -> Dict[str, Any]:
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
    def from_path(cls, path: PathLike, *, relation_code: Optional[str] = None, lazy: bool = True) -> "Relation":
        """
        Loads a relation from a JSONL file and associated metadata.

        Parameters:
            path (str): The path to the dataset directory.
            relation_code (str): The specific code of the relation to load.
            lazy (bool): If False, the instance table is loaded directly into memory.

        Returns:
            Relation: An instance of the Relation class populated with data from the file.

        Raises:
            Exception: If there is an error in loading the file or processing the data.
        """
        if relation_code is not None:
            dataset_path = Path(path)
            relation_path = cls.path_for_code(dataset_path, relation_code)
        else:
            relation_path = Path(path)
            relation_code = cls.code_from_path(relation_path)
            dataset_path = relation_path.parent

        log.debug("Loading relation from: %s", relation_path)

        with open(dataset_path / "metadata_relations.json") as meta_file:
            metadata = json.load(meta_file)[relation_code]
            templates = metadata.pop("templates", [])

        answer_space = cls.answer_space_from_metadata(metadata, id_prefix=f"{relation_code}-")

        if len(metadata) > 0:
            log.info(
                "Found additional metadata thats is going to be ignored: %s",
                ", ".join(f"{m}" for m in metadata.keys()),
            )

        if lazy:
            instance_table = None
            lazy_load_path = relation_path
        else:
            instance_table = cls.load_instance_table(relation_path, answer_space=answer_space)
            lazy_load_path = None

        return cls(
            relation_code,
            answer_space=answer_space,
            templates=templates,
            instance_table=instance_table,
            lazy_load_path=lazy_load_path,
        )

    def copy(self, **kw):
        kw = {
            "templates": self.templates.copy(),
            **kw,
        }
        return super().copy(**kw)

    def filter_subset(
        self, indices: Sequence[int], *, save_path: Optional[PathLike], keep_answer_space: bool = False
    ) -> Self:
        original_instance_table = self.instance_table
        instance_table = original_instance_table.iloc[indices].copy()

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

        relation = self.copy(
            answer_space=answer_space,
            instance_table=instance_table,
            lazy_load_path=self.path_for_code(Path(save_path), self.relation_code) if save_path is not None else None,
        )

        if save_path is not None:
            relation.save(save_path)
            relation._instance_table = None

        return relation

    @classmethod
    def load_instance_table(cls, path: Path, *, answer_space: Optional[pd.Series] = None) -> pd.DataFrame:
        instance_table = super().load_instance_table(path, answer_space=answer_space)

        if "obj_id" in instance_table and "answer_idx" not in instance_table:
            if answer_space is None:
                answer_space = cls.answer_space_from_instance_table(instance_table)
            answer_indices = pd.Series(np.arange(len(answer_space)), index=answer_space.index, name="answer_idx")
            instance_table = instance_table.join(answer_indices, on="obj_id")
        return instance_table


class Dataset(DatasetBase[Relation]):
    """
    A collection of relations forming a multiple choice dataset.

    Attributes:
        relations (List[Relation]): A list of Relation instances in the dataset.
        dataset_name (str, optional): The name of the dataset.

    Methods:
        load_from_path: Class method to load a dataset from a specified path.
    """

    def __init__(self, relations: List[Relation], path: PathLike, name: Optional[str] = None):
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
    def from_path(cls, path: PathLike, *, lazy: bool = True, **kwargs) -> "Dataset":
        """
        Loads a multiple choice dataset from a specified directory path.

        This method scans the directory for relation files and assembles them into a MultipleChoiceDataset.

        Parameters:
            path (str): The directory path where the dataset is stored.
            lazy (bool): If False, the instance tables of all relations are directly loaded into memory.

        Returns:
            Dataset: An instance of MultipleChoiceDataset loaded with the relations from the directory.

        Raises:
            Exception: If there is an error in loading the dataset.

        Usage:
            Loading the BEAR-dataset.
            ``` python
            >>> from lm_pub_quiz import Dataset
            >>> dataset = Dataset.load_from_path('/path/to/dataset/BEAR')
            ```
        """
        kwargs["path"] = path
        dataset_path = Path(path)

        if not dataset_path.exists():
            msg = f"Dataset at `{dataset_path}` could not be opened: Path does not exist."
            raise RuntimeError(msg)

        relation_files = natural_sort(dataset_path.glob("*.jsonl"))
        relations = [Relation.from_path(p, lazy=lazy) for p in relation_files]

        # if no name was passed, default to using the name of the dataset directory
        if "name" not in kwargs:
            kwargs["name"] = dataset_path.stem

        log.info("Loaded dataset `%s` (%d relations) from `%s`.", kwargs["name"], len(relation_files), dataset_path)

        return cls(relations, **kwargs)

    def activated(self):
        if not self.is_lazy:
            return self

        return self.__class__(name=self.name, path=self.path, relations=[rel.activated() for rel in self])

    def filter_subset(
        self,
        indices: Mapping[str, Sequence[int]],
        *,
        save_path: Optional[PathLike] = None,
        dataset_name: Optional[str] = None,
        keep_answer_space: bool = False,
    ):
        return self.__class__(
            [
                self[key].filter_subset(
                    value,
                    save_path=save_path,
                    keep_answer_space=keep_answer_space,
                )
                for key, value in indices.items()
            ],
            name=dataset_name if dataset_name is not None else f"{self.name} (subset)",
            path=save_path if save_path is not None else Path("."),
        )
