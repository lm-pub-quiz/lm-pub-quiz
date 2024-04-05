"""Results Module"""

import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd
from typing_extensions import Self

from lm_pub_quiz.data.base import DatasetBase, NoInstanceTableError, RelationBase
from lm_pub_quiz.metrics.base import RelationMetric
from lm_pub_quiz.util import PathLike, parse_dumped_raw_results

log = logging.getLogger(__name__)


class RelationResult(RelationBase):
    _instance_table_file_name_suffix = "_results.jsonl"
    _metadata_file_name: str = "metadata_results.json"

    _default_reductions: ClassVar[Dict[str, Callable[[List[float]], float]]] = {
        "sum": cast(Callable[[Iterable[float]], float], np.sum),
        "mean": cast(Callable[[Iterable[float]], float], np.mean),
    }

    def __init__(
        self,
        relation_code: str,
        *,
        metadata: Dict[str, Any],
        metric_values: Optional[Dict[str, Any]] = None,
        instance_table: Optional[pd.DataFrame] = None,
        answer_space: Optional[pd.Series] = None,
        lazy_load_path: Optional[Path] = None,
        lazy_include_answers: bool = True,
    ):
        super().__init__(
            relation_code, instance_table=instance_table, answer_space=answer_space, lazy_load_path=lazy_load_path
        )
        self.metadata = metadata
        self.metric_values: Dict[str, Any] = metric_values or {}
        self._lazy_include_answers = lazy_include_answers

    def get_metadata(self) -> Dict[str, Any]:
        return {
            **self.metadata,
            "metric_values": self.metric_values,
            **super().get_metadata(),
        }

    @property
    def has_instance_table(self):
        return self._instance_table is not None or self._lazy_load_path is not None

    @property
    def used_template(self):
        return self.metadata["templates"][self.metadata["template_index"]]

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        relation_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        lazy: bool = True,
        include_answers: bool = True,
    ) -> "RelationResult":
        """
        Loads the evaluated relation from a JSONL file and associated metadata.

        Parameters:
            path (PathLike): The path to the relations instance table.

        Returns:
            RelationResult: An instance of the RelationResult class populated with data from the file.

        Raises:
            Exception: If there is an error in loading the file or processing the data.
        """
        path_to_load: Path = Path(path)

        if not path_to_load.exists():
            msg = f"Could not load {cls.__name__} ({relation_code}): Path `{path_to_load}` not found."
            raise FileNotFoundError(msg)

        if relation_code is None:
            if path_to_load.is_file():
                relation_code = cls.code_from_path(path_to_load)
            else:
                msg = (
                    f"Could not load {cls.__name__} ({relation_code}): "
                    f"Path `{path_to_load}` found, but unclear which relation to load."
                )
                raise RuntimeError(msg)
        elif path_to_load.is_dir():
            path_to_load = cls.path_for_code(path_to_load, relation_code)

        try:
            if metadata is None:
                metadata_path = path_to_load.parent / "metadata_results.json"
                if metadata_path.exists():
                    with open(metadata_path) as meta_file:
                        metadata = json.load(meta_file)[relation_code]
                else:
                    metadata = {}

            assert metadata is not None

            answer_space = cls.answer_space_from_metadata(metadata, id_prefix=f"{relation_code}-")

            obj = cls(
                relation_code=relation_code,
                metric_values=metadata.pop("metric_values", None),
                metadata=metadata,
                answer_space=answer_space,
                lazy_load_path=path_to_load if path_to_load.exists() else None,
                lazy_include_answers=include_answers,
            )

            if not lazy and path_to_load is not None:
                try:
                    obj._instance_table = obj.load_instance_table(path_to_load, answer_space=answer_space)
                except NoInstanceTableError as e:
                    log.warning(str(e))

            return obj

        except KeyError:
            log.error(
                "There is no metadata for given result with relation code %s, even though the metadata file exists.",
                relation_code,
            )
            raise
        except Exception as e:
            log.error("Failed to load relation result from file: %s.", path_to_load)
            log.exception(e)
            raise

    def copy(self, **kw):
        kw = {"metadata": deepcopy(self.metadata), **kw}
        return super().copy(**kw)

    def reduced(
        self,
        reduction: Union[str, Callable] = "sum",
        *,
        save_path: Optional[PathLike] = None,
        reduction_name: Optional[str] = None,
        pass_indices: bool = False,
    ) -> Self:
        if self.metadata.get("reduction") is not None:
            msg = "The conversion to reduced format is only possible if the original reduction is set to None."
            raise RuntimeError(msg)

        if reduction_name is None:
            if isinstance(reduction, str):
                reduction_name = reduction
            else:
                reduction_name = reduction.__name__

        if isinstance(reduction, str):
            reduction_func: Callable = self._default_reductions[reduction]
        else:
            reduction_func = reduction

        instance_table = self.instance_table

        if not pass_indices:
            reduced_scores = instance_table["pll_scores"].map(lambda values: [reduction_func(v) for v in values])
        else:
            reduced_scores = instance_table.apply(
                lambda row: [
                    reduction_func(scores, sub=sub, obj=obj, template=template)
                    for scores, sub, obj, template in zip(
                        row["pll_scores"], row["sub_indices"], row["obj_indices"], row["template_indices"]
                    )
                ],
                axis=1,
            )

        instance_table = instance_table.copy()
        instance_table.drop(columns=["tokens", "sub_indices", "obj_indices", "template_indices"], inplace=True)
        instance_table["pll_scores"] = reduced_scores

        metadata = self.metadata.copy()
        metadata["reduction"] = reduction_name

        rel = self.__class__(
            relation_code=self.relation_code,
            metadata=metadata,
            instance_table=instance_table,
            answer_space=self.answer_space,
        )

        if save_path is not None:
            rel._lazy_load_path = rel.save(save_path)

        return rel

    def get_metric(self, metric: str):
        rel = self
        if self.metadata["reduction"] is None:
            warnings.warn(
                "Results are in the raw form. Defaulting to 'sum' reduction. "
                "Use `raw_format_converter` method for the explicit choice of the reduction method.",
                stacklevel=1,
            )

            rel = self.reduced(reduction="sum")

        if metric in rel.metric_values:
            return rel.metric_values[metric]

        else:
            instance_table = rel.instance_table

            if instance_table is not None:
                metric_obj = RelationMetric.create_metric(metric)
                metric_obj.reset()
                metric_obj.add_instances_from_table(instance_table)

                self.metric_values.update(metric_obj.compute())

                return self.metric_values[metric]
            else:
                msg = f"Could not compute metric `{metric}`: Metric not precomputed and no instance table found."
                raise RuntimeError(msg)

    def __len__(self) -> int:
        if self.is_lazy and "num_instances" in self.metadata:
            return self.metadata["num_instances"]
        else:
            return len(self.instance_table)

    def filter_subset(
        self,
        indices: Sequence[int],
        *,
        save_path: Optional[PathLike] = None,
        keep_answer_space: bool = False,
        dataset_name: Optional[str] = None,
    ) -> Self:
        original_instance_table = self.instance_table
        original_answer_space = self.answer_space

        instance_table = original_instance_table.iloc[indices].copy()

        metadata = self.metadata.copy()
        if dataset_name is not None:
            metadata["dataset_name"] = dataset_name

        if keep_answer_space:
            answer_space = original_answer_space
        else:
            answer_space = self.answer_space_from_instance_table(instance_table)

            old_positions = [self.answer_space.index.get_loc(_id) for _id in answer_space.index]

            answer_space_translation = {_id: i for i, _id in enumerate(old_positions)}

            if self.metadata["reduction"] is None:

                def _filter_indices(values, kept_indices=old_positions):
                    return [values[i] for i in kept_indices]

                columns = ["tokens", "pll_scores", "sub_indices", "obj_indices", "template_indices"]
                instance_table[columns] = instance_table[columns].applymap(_filter_indices)

            else:

                def _filter_scores(scores):
                    return [scores[i] for i in old_positions]

                instance_table["pll_scores"] = instance_table["pll_scores"].map(_filter_scores)

            instance_table["answer_idx"] = instance_table["answer_idx"].map(answer_space_translation)

        rel = self.__class__(
            relation_code=self.relation_code,
            instance_table=instance_table,
            metadata=metadata,
            answer_space=answer_space,
        )

        if save_path is not None:
            rel._lazy_load_path = rel.save(save_path)
            rel._instance_table = None

        return rel

    @classmethod
    def load_instance_table(cls, path: Path, *, answer_space: Optional[pd.Series] = None) -> pd.DataFrame:
        instance_table = super().load_instance_table(path)

        if "answer_idx" not in instance_table:
            # Legacy format -> convert to new

            if answer_space is None:
                if "obj_id" not in instance_table:
                    msg = "Object id not included in the instance table. Cannot load this format."
                    raise ValueError(msg)
                elif "obj_label" not in instance_table:
                    msg = "Cannot derive correct answer index since object labels are not included."
                    raise ValueError(msg)
                else:
                    answer_space = cls.answer_space_from_instance_table(instance_table)

            if "results" in instance_table:
                # Non-reduced format
                answer_space = cls.answer_space_from_instance_table(instance_table)
                instance_table["answer_idx"] = instance_table["obj_id"].apply(lambda x: answer_space.index.get_loc(x))

                duplicated = answer_space.duplicated()
                if duplicated.any():
                    log.info(
                        "Found duplicate labels in answer space:\n%s",
                        "\n".join(f"{i} - {v}" for i, v in answer_space[duplicated].items()),
                    )

                    if (instance_table["results"].map(len) == len(answer_space)).all():
                        log.info("Number of predictions matches number of answer ids.")

                    elif (instance_table["results"].map(len) == len(answer_space) - duplicated.sum()).all():
                        warnings.warn(
                            "Number of predictions matches number of answer labels: fixing the predictions.",
                            stacklevel=1,
                        )

                        label_indices = {label: i for i, label in enumerate(answer_space.unique())}

                        def _pred_expansion(r):
                            return [r[label_indices[label]] for label in answer_space]

                        instance_table["results"] = instance_table["results"].map(_pred_expansion)

                    else:
                        warnings.warn("Inconsistent number of predictions", stacklevel=1)

                new_representation = instance_table["results"].apply(lambda x: parse_dumped_raw_results(x))
                new_columns = ["tokens", "pll_scores", "sub_indices", "obj_indices", "template_indices"]
                instance_table[new_columns] = pd.DataFrame(new_representation.tolist(), index=instance_table.index)

                # account for suprisal -> pll scores
                instance_table["pll_scores"] = instance_table["pll_scores"].apply(
                    lambda x: [[-item for item in lst] for lst in x]
                )

                instance_table.drop(
                    columns=["sub_aliases", "results", "predicted_tokens"], inplace=True, errors="ignore"
                )

            else:
                # Reduced format
                answer_indices = pd.Series(np.arange(len(answer_space)), index=answer_space.index, name="answer_idx")

                log.debug(answer_indices)
                log.debug(instance_table)

                instance_table = instance_table.join(answer_indices, on="obj_id")

                # unsort and invert pll scores
                new_pll_scores = []

                for _, row in instance_table.iterrows():
                    answer_scores = dict(zip(row["predicted_tokens"], row["pll_scores"]))

                    new_pll_scores.append([-answer_scores[label] for label in answer_space])

                instance_table["pll_scores"] = pd.Series(new_pll_scores)

                instance_table.drop(columns=["predicted_tokens"], inplace=True, errors="ignore")

        elif "obj_id" not in instance_table or "obj_label" not in instance_table:
            if "answer_idx" not in instance_table:
                msg = f"No object or answer index found in instance table: {path}."
                raise RuntimeError(msg)

            if answer_space is None:
                msg = "Answers cannot be included since no answer space was given."
                raise ValueError(msg)

            instance_table = instance_table.join(answer_space.to_frame().reset_index(), on="answer_idx")

        return instance_table

    @classmethod
    def save_instance_table(cls, instance_table: pd.DataFrame, path: Path):
        instance_table = instance_table.drop(
            columns=["obj_id", "obj_label"], errors="ignore"
        )  # ignore non-existing columns
        super().save_instance_table(instance_table, path)

    @property
    def relation_type(self) -> str:
        if "relation_type" in self.metadata:
            return self.metadata["relation_type"]
        elif self.instance_table.duplicated("obj_id").any():
            return "multiple instances per answer"
        else:
            return "single instance per answer"


class DatasetResults(DatasetBase[RelationResult]):
    """Container for relation results."""

    def __init__(self, results: Optional[List[RelationResult]] = None):
        self.relation_data = results or []

    def append(self, result: RelationResult) -> None:
        self.relation_data.append(result)

    @classmethod
    def from_path(cls, path: PathLike, *, lazy: bool = True, **kwargs) -> "DatasetResults":
        """
        Loads a results from a specified directory path.

        This method scans the directory for relation files and assembles them into a DatasetResults.

        Parameters:
            path (str): The directory path where the dataset is stored.

        Returns:
            DatasetResults: An instance of DatasetResults loaded with the results from the directory.

        Raises:
            Exception: If there is an error in loading the dataset.

        Usage:
            Loading all relation results for a dataset.
            ``` python
            from results import DatasetResults
            results = DatasetResults.load_from_path('/path/to/results/', dataset_name='BEAR')
            ```
        """
        results_path = Path(path)

        if not results_path.exists():
            msg = f"{cls.__name__} at `{results_path}` could not be opened: Path does not exist."
            raise FileNotFoundError(msg)

        if results_path.is_dir():
            metadata_path = results_path / "metadata_results.json"

            if not results_path.exists():
                msg = f"Metadata for dataset at `{results_path}` not found (expected at `{metadata_path})."
                raise FileNotFoundError(msg)
        else:
            metadata_path = results_path

        log.info("Loading results from: %s", results_path)

        with open(metadata_path) as meta_file:
            dataset_metadata = json.load(meta_file)

        obj = cls([], **kwargs)

        for relation_code, relation_metadata in dataset_metadata.items():
            obj.append(
                RelationResult.from_path(
                    metadata_path.parent,
                    metadata=relation_metadata,
                    relation_code=relation_code,
                    lazy=lazy,
                )
            )
        return obj

    def activated(self) -> Self:
        if not self.is_lazy:
            return self
        else:
            return self.__class__([rel.activated() for rel in self])

    @staticmethod
    def _construct_metrics_dict(metrics: Iterable[str], relation: RelationResult) -> Dict[str, float]:
        d = {}
        for m in metrics:
            metric_value = relation.get_metric(m)
            if isinstance(metric_value, dict):
                d.update({f"{m}_{k}": v for k, v in metric_value.items()})
            else:
                d[m] = metric_value

        return d

    def get_metrics(
        self,
        metrics: Union[str, Iterable[str]],
        *,
        accumulate_all: bool = False,
        accumulate_relation_types: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        if accumulate_relation_types and accumulate_all:
            warnings.warn(
                "It is recommended to only set `accumulate_all` or `accumulate_relation_types`. "
                "Defaulting to accumultate for each relation type.",
                stacklevel=1,
            )

        if isinstance(metrics, str):
            metrics = [metrics]

        if (accumulate_all or accumulate_relation_types) and "num_instances" not in metrics:
            metrics = [*metrics, "num_instances"]

        df = pd.DataFrame({rel.relation_code: self._construct_metrics_dict(metrics, rel) for rel in self}).T

        df["relation_type"] = pd.Series({rel.relation_code: rel.relation_type for rel in self})

        if accumulate_all or accumulate_relation_types:

            def acc_group(group):
                return pd.Series(
                    {
                        k: (np.average(v, weights=group["num_instances"]) if k != "num_instances" else v.sum())
                        for k, v in group.items()
                        if k != "relation_type"
                    }
                )

            if accumulate_relation_types:
                return pd.concat(
                    [
                        df.groupby("relation_type").apply(acc_group),
                        pd.DataFrame([acc_group(df).to_dict()], index=["all"]),
                    ]
                )
            else:
                return acc_group(df)
        else:
            return df

    def filter_subset(
        self,
        indices: Mapping[str, Sequence[int]],
        *,
        save_path: Optional[PathLike] = None,
        keep_answer_space: bool = False,
        dataset_name: Optional[str] = None,
    ):
        return self.__class__(
            [
                self[key].filter_subset(
                    value, save_path=save_path, keep_answer_space=keep_answer_space, dataset_name=dataset_name
                )
                for key, value in indices.items()
            ]
        )

    def reduced(
        self,
        reduction: Union[str, Callable],
        *,
        save_path: Optional[PathLike] = None,
        reduction_name: Optional[str] = None,
        pass_indices: bool = False,
    ) -> Self:

        converted_results = [
            rel.reduced(
                reduction=reduction, save_path=save_path, reduction_name=reduction_name, pass_indices=pass_indices
            )
            for rel in self
        ]

        return self.__class__(results=converted_results)

    @overload
    def get_metadata(self, key: None = None) -> Dict[str, Any]: ...

    @overload
    def get_metadata(self, key: str) -> Any: ...

    @overload
    def get_metadata(self, key: List[str]) -> Dict[str, Any]: ...

    def get_metadata(self, key: Optional[Union[str, List[str]]] = None):
        """Return metadata from the relations. If no keys are passed, all consistent values are returned."""

        intersection: Any = None
        _is_first: bool = True

        for rel in self:
            m = rel.get_metadata()

            if _is_first:
                if key is None:
                    intersection = {**m}
                elif isinstance(key, str):
                    intersection = m[key]
                else:
                    intersection = {k: m[k] for k in key if k in m}

                _is_first = False

            elif isinstance(key, str):
                if key not in m or m[key] != intersection:
                    intersection = None
            else:
                for k, v in m.items():
                    if k in intersection and v != intersection[k]:
                        del intersection[k]

        return intersection
