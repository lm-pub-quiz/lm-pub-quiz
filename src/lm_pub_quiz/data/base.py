import json
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, Iterator, List, Mapping, Optional, Sequence, TypeVar, Union, cast

import pandas as pd
from typing_extensions import Self

from lm_pub_quiz.util import PathLike

log = logging.getLogger(__name__)


class NoInstanceTableError(Exception):
    pass


class DataBase(ABC):
    """Base class for the representation of relations, relations results, and dataset collections."""

    @classmethod
    @abstractmethod
    def from_path(cls, path: PathLike, *, lazy: bool = True):
        """Load data from the given path.

        If `lazy`, only the metadata is loaded and the instances are loaded once they are accessed.
        """
        pass

    @abstractmethod
    def save(self, path: PathLike):
        """Save the data under the given path."""

        pass

    @property
    @abstractmethod
    def is_lazy(self) -> bool:
        """Return true if lazy loading is active."""
        pass

    @abstractmethod
    def activated(self) -> Self:
        """Return self if lazy loading is active, otherwise return a copy without lazy loading."""
        pass


class RelationBase(DataBase):
    """Base class for the representation of relations and relations results."""

    _instance_table_file_name_suffix: str = ".jsonl"
    _metadata_file_name: str = "metadata_relations.json"

    _len: Optional[int] = None

    def __init__(
        self,
        relation_code: str,
        *,
        lazy_load_path: Optional[Path] = None,
        instance_table: Optional[pd.DataFrame] = None,
        answer_space: Optional[pd.Series] = None,
    ):

        self._relation_code = relation_code
        self._lazy_load_path = lazy_load_path
        self._instance_table = instance_table
        self._answer_space = answer_space

    @property
    def relation_code(self) -> str:
        return self._relation_code

    def copy(self, **kw):
        """Create a copy of the isntance with specified fields replaced by new values."""

        kw = {
            "relation_code": self.relation_code,
            "lazy_load_path": self._lazy_load_path,
            "instance_table": self._instance_table.copy() if self._instance_table is not None else None,
            "answer_space": self._answer_space.copy() if self._answer_space is not None else None,
            **kw,
        }
        return self.__class__(kw.pop("relation_code"), **kw)

    def activated(self) -> Self:
        """Return self or a copy of self with the instance_table loaded (lazy loading disabled)."""

        if not self.is_lazy:
            return self

        return self.copy(instance_table=self.instance_table)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} `{self.relation_code}`"

    def get_metadata(self) -> Dict[str, Any]:
        if self._answer_space is not None:
            return {
                "answer_space_labels": self.answer_space.tolist(),
                "answer_space_ids": self.answer_space.index.tolist(),
            }
        else:
            return {}

    @staticmethod
    def _generate_obj_ids(n: int, *, id_prefix: str = ""):
        return id_prefix + pd.RangeIndex(n, name="obj_id").astype(str)

    @classmethod
    def answer_space_from_instance_table(cls, instance_table: pd.DataFrame, **kw) -> pd.Series:
        if "obj_label" not in instance_table:
            msg = "Cannot generate answer space: No object information in instance table."
            raise ValueError(msg)

        if "obj_id" in instance_table:
            answer_groups = instance_table.groupby("obj_id", sort=False).obj_label
            unique_ids = answer_groups.nunique().eq(1)

            if not unique_ids.all():
                ids = ", ".join(f"'{v}'" for v in unique_ids[~unique_ids].index)
                log.warning("Some object IDs contain multiple labels: %s", ids)

            return answer_groups.first()

        else:
            answer_labels = instance_table["obj_label"].unique()
            return pd.Series(answer_labels, index=cls._generate_obj_ids(len(answer_labels), **kw), name="obj_label")

    @classmethod
    def answer_space_from_metadata(cls, metadata, **kw) -> Optional[pd.Series]:
        if "answer_space_labels" in metadata and "answer_space_ids" in metadata:
            if "answer_space_labels" in metadata:
                answer_space_labels = metadata.pop("answer_space_labels")
            else:
                answer_space_labels = metadata.pop("answer_space")

            answer_space_ids = metadata.pop("answer_space_ids", None)

            if answer_space_ids is None:
                answer_space_ids = cls._generate_obj_ids(len(answer_space_labels), **kw)

            index = pd.Index(answer_space_ids, name="obj_id")

            answer_space = pd.Series(answer_space_labels, index=index, name="obj_label")

            return answer_space
        elif (
            "answer_space_labels" not in metadata
            and "answer_space_ids" not in metadata
            and "answer_space" not in metadata
        ):
            return None
        else:
            warnings.warn(
                "To define an answer space in the medata data, specify `answer_space_ids` and "
                "`answer_space_labels` (using answer space base on the instance table).",
                stacklevel=1,
            )
            return None

    @property
    def answer_space(self) -> pd.Series:
        if self._answer_space is None:
            # invoke file loading to get answer space
            _ = self.instance_table

        return cast(pd.Series, self._answer_space)

    @property
    def instance_table(self) -> pd.DataFrame:
        if self._instance_table is None:
            if self._lazy_load_path is None:
                msg = (
                    f"Could not load instance table for {self.__class__.__name__} "
                    f"({self.relation_code}): No path given."
                )
                raise NoInstanceTableError(msg)

            instance_table = self.load_instance_table(self._lazy_load_path, answer_space=self._answer_space)

            if self._answer_space is None:
                # store answer_space
                self._answer_space = self.answer_space_from_instance_table(
                    instance_table, id_prefix=f"{self.relation_code}-"
                )

            # store number of instances
            self._len = len(instance_table)

            return instance_table

        return self._instance_table

    def __len__(self) -> int:
        if self._instance_table is None:
            if self._len is None:
                # invoke file loading to get answer space
                _ = self.instance_table
            return cast(int, self._len)
        else:
            return len(self.instance_table)

    @abstractmethod
    def filter_subset(
        self, indices: Sequence[int], *, save_path: Optional[PathLike], keep_answer_space: bool = False
    ) -> Self:
        pass

    @classmethod
    def load_instance_table(
        cls, path: Path, *, answer_space: Optional[pd.Series] = None  # noqa: ARG003
    ) -> pd.DataFrame:
        if not path.exists():
            msg = f"Could not load instance table for {cls.__name__}: Path `{path}` could not be found."
            raise FileNotFoundError(msg)
        elif not path.is_file():
            msg = f"Could not load instance table for {cls.__name__}: `{path}` is not a file."
            raise RuntimeError(msg)

        log.debug("Loading instance table from: %s", path)
        instance_table = pd.read_json(path, lines=True)

        return instance_table

    @classmethod
    def save_instance_table(cls, instance_table: pd.DataFrame, path: Path):
        instance_table.to_json(path, orient="records", lines=True)

    @property
    def is_lazy(self) -> bool:
        return self._instance_table is None and self._lazy_load_path is not None

    @property
    @abstractmethod
    def has_instance_table(self) -> bool:
        pass

    def save(self, save_path: PathLike) -> Optional[Path]:
        """Save results to a file and export meta_data"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        log.debug("Saving %s result to: %s", self, save_path)

        ### Metadata file -> .json ###
        if save_path.is_dir():
            metadata_path = save_path / self._metadata_file_name
        else:
            metadata_path = save_path
            save_path = save_path.parent

        if metadata_path.exists():
            with open(metadata_path) as file:
                all_metadata = json.load(file)

                if self.relation_code in all_metadata:
                    log.warning("Overwriting metadata info for relation %s (%s)", self.relation_code, save_path)
        else:
            all_metadata = {}

        all_metadata[self.relation_code] = self.get_metadata()

        with open(metadata_path, "w") as file:
            json.dump(all_metadata, file, indent=4, default=str)
            log.debug("Metadata file was saved to: %s", metadata_path)

        ### Store instance table to .jsonl file ###
        if self.has_instance_table:
            instances_path = self.path_for_code(save_path, self.relation_code)
            self.save_instance_table(self.instance_table, instances_path)
            log.debug("Instance table was saved to: %s", instances_path)

            return instances_path

        return None

    @classmethod
    def code_from_path(cls, path: Path) -> str:
        if not path.name.endswith(cls._instance_table_file_name_suffix):
            msg = (
                f"Incorrect path for {cls.__name__} instance table "
                f"(expected suffix {cls._instance_table_file_name_suffix}): {path}"
            )
            raise ValueError(msg)
        return path.name[: -len(cls._instance_table_file_name_suffix)]

    @classmethod
    def path_for_code(cls, path: Path, relation_code: str) -> Path:
        return path / f"{relation_code}{cls._instance_table_file_name_suffix}"


RT = TypeVar("RT", bound=RelationBase)


class DatasetBase(DataBase, Generic[RT]):
    """Base class for a collection of relations or relations results."""

    relation_data: List[RT]

    def __len__(self) -> int:
        return len(self.relation_data)

    def __str__(self) -> str:
        relations = ", ".join(self.relation_codes)
        return f"{self.__class__.__name__}({relations})"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key: Union[int, str]) -> RT:
        if isinstance(key, int):
            return self.relation_data[key]
        else:
            for relation in self:
                if relation.relation_code == key:
                    return relation

            # no match
            msg = f"Relation {key} not found in this {self.__class__.__name__}."
            raise KeyError(msg)

    def __iter__(self) -> Iterator[RT]:
        yield from self.relation_data

    @abstractmethod
    def filter_subset(
        self,
        indices: Mapping[str, Sequence[int]],
        *,
        save_path: Optional[PathLike] = None,
        keep_answer_space: bool = False,
        dataset_name: Optional[str] = None,
    ):
        pass

    @property
    def relation_codes(self) -> Iterable[str]:
        return (r.relation_code for r in self)

    @property
    def is_lazy(self) -> bool:
        return any(rel.is_lazy for rel in self)

    def save(self, path: PathLike) -> None:
        for result in self:
            result.save(path)
