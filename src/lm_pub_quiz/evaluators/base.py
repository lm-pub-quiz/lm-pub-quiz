"""MultipleChoiceEvaluator"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import (
    Any,
    Optional,
    Union,
    cast,
)

import pandas as pd
from tqdm.auto import tqdm

from lm_pub_quiz.__about__ import __version__
from lm_pub_quiz.data import Dataset, DatasetResults, Relation, RelationResult
from lm_pub_quiz.data.base import InstanceTableFileFormat
from lm_pub_quiz.evaluators.model_util import ModelMixin
from lm_pub_quiz.evaluators.templating_util import Templater
from lm_pub_quiz.metrics import RelationMetric
from lm_pub_quiz.metrics.base import MetricSpecification
from lm_pub_quiz.types import (
    EachTokenReturnFormat,
    PathLike,
    ReducedReturnFormat,
)
from lm_pub_quiz.util import parse_dumped_raw_results

tqdm.pandas()

log = logging.getLogger(__name__)


MultiMetricSpecification = Union[MetricSpecification, Sequence[MetricSpecification]]


class BaseEvaluator(Templater, ModelMixin, ABC):
    default_reduction = "sum"

    def evaluate_dataset(
        self,
        dataset: Dataset,
        template_index: int = 0,
        *,
        batch_size: int = 1,
        subsample: Optional[int] = None,
        save_path: Optional[PathLike] = None,
        fmt: InstanceTableFileFormat = None,
        reduction: Optional[str] = "default",
        create_instance_table: bool = True,
        metric: Optional[MultiMetricSpecification] = None,
    ) -> DatasetResults:
        """Evaluate the model on all relations in the dataset."""
        if reduction == "default":
            reduction = self.default_reduction

        if not create_instance_table and metric is None:
            msg = "Neither the instance table nor any metrics are computed: Specify the use of at least one."
            raise ValueError(msg)

        dataset_results = DatasetResults()

        log.debug("Evaluating `%s` on `%s`", self.model_name, dataset.name)

        for relation in tqdm(dataset, total=len(dataset), unit="relations", desc=f"Dataset {dataset.name}"):
            try:
                log.info("Evaluating `%s` on %s.", self.model_name, relation)
                relation_result = self.evaluate_relation(
                    relation,
                    template_index=template_index,
                    subsample=subsample,
                    reduction=reduction,
                    create_instance_table=create_instance_table,
                    metric=metric,
                    batch_size=batch_size,
                )
                relation_result._metadata.update(self.get_result_metadata(dataset=dataset))

                if save_path is not None:
                    log.debug("Saving")
                    relation_result = relation_result.saved(save_path, fmt=fmt)

                dataset_results.append(relation_result)

            except RuntimeError as e:
                logging.error("Encountered RuntimeException while evaluating `%s` on %s.", self.model_name, relation)
                log.exception(e)
                log.warning("Continuing execution (you may want to rerun relation %s)...", relation)
                continue
        log.info("Completed the evaluation on dataset `%s`.", dataset.name)

        return dataset_results

    def evaluate_relation(
        self,
        relation: Relation,
        template_index: int = 0,
        *,
        batch_size: int = 1,
        subsample: Optional[int] = None,
        reduction: Optional[str] = "default",
        create_instance_table: bool = True,
        metric: Optional[MultiMetricSpecification] = None,
    ) -> RelationResult:
        if reduction == "default":
            reduction = self.default_reduction

        instances = relation.instance_table if subsample is None else relation.subsample(subsample)

        relation_result = RelationResult(
            relation_code=relation.relation_code,
            instance_table=None,
            answer_space=relation.answer_space.copy(),
            metadata={
                "templates": relation.templates,
                "template_index": template_index,
                "model_name_or_path": self.model_name,
                "num_original_instances": len(instances),
                "subsampled": subsample,
                "time_start": datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            },
            relation_info=relation.relation_info(),
        )
        template = relation.templates[template_index]

        evaluated_instances: list[dict] = []

        metrics: list[RelationMetric] = []
        if metric is not None:
            if isinstance(metric, (str, RelationMetric)) or (
                isinstance(metric, type) and issubclass(metric, RelationMetric)
            ):
                metric = (metric,)

            for m in metric:
                m_obj: RelationMetric = RelationMetric.create_metric(m)
                m_obj.reset()
                metrics.append(m_obj)

        for _, r in tqdm(
            instances.iterrows(),
            total=len(instances),
            desc=f"Relation {relation.relation_code}",
        ):
            row = r.to_dict()

            pll_scores = self.evaluate_instance(
                template=template,
                answers=relation.answer_space.tolist(),
                subject=str(row["sub_label"]),
                reduction=reduction,
                batch_size=batch_size,
            )

            if reduction is None:
                row["tokens"], row["pll_scores"], row["sub_indices"], row["obj_indices"], row["template_indices"] = (
                    parse_dumped_raw_results(cast(EachTokenReturnFormat, pll_scores))
                )
            else:
                row["pll_scores"] = pll_scores

            # update metrics with the row
            for m in metrics:
                m.add_instance(row)

            if create_instance_table:
                # add row to resulting
                evaluated_instances.append(row)

        log.debug("Creating instance table")
        if create_instance_table:
            relation_result._instance_table = pd.DataFrame(evaluated_instances, index=instances.index)

        for m in metrics:
            relation_result.metric_values.update(m.compute())

        relation_result._metadata.update(
            {
                "time_end": datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                **self.get_result_metadata(reduction=reduction),
            }
        )

        return relation_result

    @abstractmethod
    def evaluate_instance(
        self,
        *,
        template: str,
        answers: Sequence[str],
        subject: Optional[str] = None,
        reduction: Optional[str],
        batch_size: int = 1,
        print_ranking: bool = False,
    ) -> Union[ReducedReturnFormat, EachTokenReturnFormat]:
        """Return the scores for each of the answer options.


        This function needs to be implemented by each of the concrete
        Evaluator subclasses.
        """

    def get_result_metadata(self, **kw) -> dict[str, Any]:
        metadata = {
            "lm_pub_quiz_version": __version__,
        }

        if "dataset" in kw:
            dataset = kw["dataset"]
            metadata["dataset_path"] = dataset.path
            metadata["dataset_name"] = dataset.name

        if "reduction" in kw:
            metadata["reduction"] = kw["reduction"]
        return metadata

    @staticmethod
    def print_ranking(answers: Iterable[str], scores: list[float]) -> None:
        data = zip(answers, scores)
        sorted_data = sorted(data, key=lambda x: x[1], reverse=False)
        max_str_length = max([len(item[0]) for item in sorted_data])

        # Print header
        print(f"{'Rank':<5}{'Word':<{max_str_length + 2}}{'Score':<10}")  # noqa: T201
        print("-" * (max_str_length + 26))  # noqa: T201

        # Print each item
        for rank, (word, score) in enumerate(sorted_data, 1):
            print(f"{rank:<5}{word:<{max_str_length + 2}}{score:<10}")  # noqa: T201
