"""MultipleChoiceEvaluator"""

import itertools
import logging
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from typing import (
    Any,
    Literal,
    Optional,
    Union,
    overload,
)

import pandas as pd
from tqdm.auto import tqdm

from lm_pub_quiz.__about__ import __version__
from lm_pub_quiz.data import Dataset, DatasetResults, Relation, RelationResult
from lm_pub_quiz.data.base import InstanceTableFileFormat, Item
from lm_pub_quiz.metrics import RelationMetric
from lm_pub_quiz.metrics.base import MetricSpecification
from lm_pub_quiz.model_interface import MODEL_INTERFACE_CLASSES, ModelInterface
from lm_pub_quiz.templating import Templater
from lm_pub_quiz.types import (
    ItemScores,
    ItemTokenScoresAndRoles,
    PathLike,
    TextRoles,
)
from lm_pub_quiz.util import parse_dumped_raw_results, tee_unzip

tqdm.pandas()

log = logging.getLogger(__name__)


MultiMetricSpecification = Union[MetricSpecification, Sequence[MetricSpecification]]


class Evaluator:
    model_interface: ModelInterface

    def __init__(self, *, model_interface: ModelInterface, templater: Optional[Templater] = None):
        self.model_interface = model_interface

        if templater is not None:
            self.templater = templater
        else:
            self.templater = Templater()

    @classmethod
    def from_model(cls, model, *, model_interface: str = "hf", templater: Optional[Templater] = None, **kw):
        try:
            interface_cls = MODEL_INTERFACE_CLASSES[model_interface.lower()]
            return cls(
                model_interface=interface_cls.from_model(model=model, **kw),
                templater=templater,
            )

        except KeyError as e:
            msg = f"Model interface '{model_interface}' not implemented."
            raise NotImplementedError(msg) from e

    def evaluate_dataset(
        self,
        dataset: Dataset,
        template_index: Union[int, Sequence[int], None] = None,
        *,
        subsample: Optional[int] = None,
        save_path: Optional[PathLike] = None,
        fmt: InstanceTableFileFormat = None,
        create_instance_table: bool = True,
        metric: Optional[MultiMetricSpecification] = None,
        **kw,
    ) -> DatasetResults:
        """Evaluate the model on all relations in the dataset."""

        if not create_instance_table and metric is None:
            msg = "Neither the instance table nor any metrics are computed: Specify the use of at least one."
            raise ValueError(msg)

        dataset_results = DatasetResults()

        log.debug("Evaluating `%s` on `%s`", self.model_interface.model_name, dataset.name)

        for relation in tqdm(dataset, total=len(dataset), unit="relations", desc=f"Dataset {dataset.name}"):
            try:
                log.info("Evaluating `%s` on %s.", self.model_interface.model_name, relation)
                relation_result = self.evaluate_relation(
                    relation,
                    template_index=template_index,
                    subsample=subsample,
                    create_instance_table=create_instance_table,
                    metric=metric,
                    **kw,
                )
                relation_result._metadata.update(self.get_result_metadata(dataset=dataset))

                if save_path is not None:
                    log.debug("Saving")
                    relation_result = relation_result.saved(save_path, fmt=fmt)

                dataset_results.append(relation_result)

            except RuntimeError as e:
                logging.error(
                    "Encountered RuntimeException while evaluating `%s` on %s.",
                    self.model_interface.model_name,
                    relation,
                )
                log.exception(e)
                log.warning("Continuing execution (you may want to rerun relation %s)...", relation)
                continue
        log.info("Completed the evaluation on dataset `%s`.", dataset.name)

        return dataset_results

    def evaluate_relation(
        self,
        relation: Relation,
        *,
        template_index: Union[int, Sequence[int], None] = None,
        subsample: Optional[int] = None,
        create_instance_table: bool = True,
        metric: Optional[MultiMetricSpecification] = None,
        **kw,
    ) -> RelationResult:
        items = relation.get_items(subsample=subsample, template_index=template_index)

        def f(item):
            log.debug("Item in relation %s: %s", relation.relation_code, str(item))
            return item

        items = map(f, items)

        item_iterators = itertools.tee(items, 2)

        item_results = self.evaluate_item(item_iterators[0], **kw)

        evaluated_items: list[dict] = []
        # Prepare metrics
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

        for item, result in zip(item_iterators[1], item_results):
            formatted_result = parse_dumped_raw_results(result)


            log.debug("Evaluated %s\n%s", str(item), "\n".join(f"- {k}: {v}" for k, v in formatted_result.items()))

            row = {**formatted_result, **item.to_dict()}

            # update metrics with the row
            for m in metrics:
                m.add_instance(row)

            if create_instance_table:
                # add row to resulting
                evaluated_items.append(row)

        relation_result = RelationResult(
            relation_code=relation.relation_code,
            instance_table=None,
            answer_space=relation.answer_space.copy(),
            metadata={
                "templates": relation.templates,
                "template_index": template_index,
                "model_name_or_path": self.model_interface.model_name,
                "num_original_instances": len(relation),
                "subsampled": subsample,
                "time_start": datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                "reduction": kw.get("reduction", "default"),
                **self.model_interface.get_metadata(),
            },
            relation_info=relation.relation_info(),
        )

        if create_instance_table:
            log.debug("Creating table with %d items.", len(evaluated_items))

            assert len(evaluated_items) > 0, "No items evaluated."

            relation_result._instance_table = pd.DataFrame(
                evaluated_items,
            )
            relation_result._instance_table.set_index(["instance_index", "template_index"])

        for m in metrics:
            relation_result.metric_values.update(m.compute())

        relation_result._metadata.update(
            {
                "time_end": datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                **self.get_result_metadata(),
            }
        )

        return relation_result

    # Single item passed
    @overload
    def evaluate_item(
        self,
        item: Item,
        *,
        template: Literal[None] = None,
        answers: Literal[None] = None,
        subject: Literal[None] = None,
        print_ranking: bool = False,
        **kw,
    ) -> Union[ItemScores, ItemTokenScoresAndRoles]: ...

    # Iterable of items passed
    @overload
    def evaluate_item(
        self,
        item: Iterable[Item],
        *,
        template: Literal[None] = None,
        answers: Literal[None] = None,
        subject: Literal[None] = None,
        print_ranking: Literal[False] = False,
        **kw,
    ) -> Union[Iterator[ItemScores], Iterator[ItemTokenScoresAndRoles]]: ...

    # No item, but template and answers passed
    @overload
    def evaluate_item(
        self,
        item: Literal[None] = None,
        *,
        template: str,
        answers: Sequence[str],
        subject: Optional[str] = None,
        print_ranking: bool = False,
        **kw,
    ) -> Union[ItemScores, ItemTokenScoresAndRoles]: ...

    def evaluate_item(
        self,
        item: Union[None, Item, Iterator[Item]] = None,
        *,
        template: Optional[str] = None,
        answers: Optional[Sequence[str]] = None,
        subject: Optional[str] = None,
        print_ranking: bool = False,
        **kw,
    ) -> Union[ItemScores, ItemTokenScoresAndRoles, Iterable[ItemScores], Iterable[ItemTokenScoresAndRoles]]:
        """Return the scores for each of the answer options.


        This function needs to be implemented by each of the concrete
        Evaluator subclasses.
        """
        if item is None:
            if template is None or answers is None:
                msg = "Either `item` or `template` and `answers` must be set."
                raise ValueError(msg)
            else:
                return self.evaluate_item(
                    Item(template=template, answers=answers, subject=subject),
                    print_ranking=print_ranking,
                    **kw,
                )
        elif template is not None or answers is not None or subject is not None:
            msg = "If `item` is set, `template`, `answers`, and `subject` cannot be set."
            raise ValueError(msg)

        elif isinstance(item, Item):
            results = next(self.evaluate_item(iter([item]), **kw))

            if print_ranking:
                self.print_ranking(item.answers, results)

            return results

        else:
            processed_items: Iterable[tuple[list[str], list[TextRoles]]] = map(self.templater.process_item, item)

            statement_options, text_roles = tee_unzip(processed_items, 2)

            return self.model_interface.score_statement_options(
                statement_options,
                text_roles=text_roles,
                **kw,
            )

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
