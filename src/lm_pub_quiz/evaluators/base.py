"""MultipleChoiceEvaluator"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
    overload,
)

import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    BatchEncoding,
    PreTrainedModel,
)

from lm_pub_quiz.__about__ import __version__
from lm_pub_quiz.data import Dataset, DatasetResults, Relation, RelationResult
from lm_pub_quiz.data.base import InstanceTableFileFormat
from lm_pub_quiz.evaluators.model_util import ModelMixin
from lm_pub_quiz.evaluators.scorers import CausalLMScorer, MaskedLMScorer, PLLScorerBase
from lm_pub_quiz.evaluators.templating_util import Templater
from lm_pub_quiz.metrics import RelationMetric
from lm_pub_quiz.metrics.base import MetricSpecification
from lm_pub_quiz.types import (
    EachTokenReturnFormat,
    PathLike,
    ReducedReturnFormat,
    ScoredToken,
    ScoringMask,
    SpanRoles,
    TokenRoles,
)
from lm_pub_quiz.util import parse_dumped_raw_results

tqdm.pandas()

log = logging.getLogger(__name__)


MultiMetricSpecification = Union[MetricSpecification, Sequence[MetricSpecification]]


class BaseEvaluator(Templater, ModelMixin, ABC):
    default_reduction = "sum"

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
        """This function should return scores for each of the answer options."""

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


class Evaluator(BaseEvaluator, PLLScorerBase):
    """PLL-based evaluator base class."""

    def __init__(
        self,
        *,
        conditional_score: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conditional_score = conditional_score

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
        if "[Y]" not in template:
            msg = 'Provided sentence is missing a placeholder ("[Y]") used for answers.'
            raise ValueError(msg)

        if reduction is None and print_ranking:
            msg = "Cannot print ranking if reduction is `None`."
            raise ValueError(msg)

        results: list = self.score_answers(
            template=template,
            answers=answers,
            reduction=reduction,
            subject=subject,
            batch_size=batch_size,
        )

        if print_ranking:
            self.print_ranking(answers, results)

        return results

    @overload
    def score_answers(
        self,
        *,
        template: str,
        answers: Sequence[str],
        reduction: None,
        subject: Optional[str] = None,
        batch_size: int = 1,
    ) -> EachTokenReturnFormat: ...

    @overload
    def score_answers(
        self,
        *,
        template: str,
        answers: Sequence[str],
        reduction: str,
        subject: Optional[str] = None,
        batch_size: int = 1,
    ) -> ReducedReturnFormat: ...

    def score_answers(
        self,
        *,
        template: str,
        answers: Sequence[str],
        reduction: Optional[str],
        subject: Optional[str] = None,
        batch_size: int = 1,
    ) -> Union[EachTokenReturnFormat, ReducedReturnFormat]:
        """Calculates sequence scores using the Casual Language Model.

        Parameters:
            template str: The template to use (should contain a `[Y]` marker).
            answers list[str]: List of answers to calculate score for.

        Returns:
            list[float]: List of suprisals scores per sequence
        """
        statements: Sequence[str]
        span_roles: Sequence[SpanRoles]

        statements, span_roles = zip(
            *(
                self.replace_placeholders(
                    template=template,
                    subject=subject,
                    answer=a,
                )
                for a in answers
            )
        )

        batch: BatchEncoding
        scoring_masks: Sequence[ScoringMask]

        batch, scoring_masks = self.encode(
            statements=statements,
            span_roles=span_roles,
        )

        token_scores: list[list[float]] = self.score_statements(
            batch, scoring_masks=scoring_masks, batch_size=batch_size
        )

        if reduction is None:
            token_roles = self.derive_token_roles(batch=batch, span_roles=span_roles, scoring_masks=scoring_masks)

            scored_tokens: list[list[ScoredToken]]

            scored_tokens = [
                list(zip(tokens, token_scores)) for tokens, token_scores in zip(self.decode_tokens(batch), token_scores)
            ]

            return list(zip(scored_tokens, token_roles))

        else:
            reduction_func = self._get_reduction_function(reduction)
            return [reduction_func(s) for s in token_scores]

    @abstractmethod
    def encode(
        self, statements: Sequence[str], span_roles: Sequence[SpanRoles]
    ) -> tuple[BatchEncoding, Sequence[ScoringMask]]:
        """Encode the statements using the tokenizer and create an appropriate scoring mask.

        In case the conditional scores need to be created, set the scoring mask accordingly.
        """

    def decode_tokens(
        self, batch: BatchEncoding, scoring_masks: Optional[Sequence[ScoringMask]] = None
    ) -> list[list[str]]:
        if scoring_masks is None:
            return [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in batch["input_ids"]]
        else:
            return [batch.tokens(i)[scoring_mask] for i, scoring_mask in enumerate(scoring_masks)]

    def derive_token_roles(
        self,
        batch: BatchEncoding,
        span_roles: Sequence[SpanRoles],
        scoring_masks: Optional[Sequence[ScoringMask]],
    ) -> Sequence[TokenRoles]:
        """Derive which tokens belong to the subject, answer, and template."""
        token_roles = []

        non_template_tokens: set[int]
        token_indices: dict[str, list[int]]

        for statement_index, spans in enumerate(span_roles):
            non_template_tokens = set()

            token_indices = {k: [] for k in spans.keys()}

            for k, v in spans.items():
                for start, end in v:
                    # go through the span until we find the first token
                    first_affected_token: Optional[int] = next(
                        (t for i in range(start, end) if (t := batch.char_to_token(statement_index, i)) is not None),
                        None,
                    )

                    # do the same, just in reversed order
                    last_affected_token: Optional[int] = next(
                        (
                            t
                            for i in reversed(range(start, end))
                            if (t := batch.char_to_token(statement_index, i)) is not None
                        ),
                        None,
                    )

                    if first_affected_token is None or last_affected_token is None:
                        # There was no token within the span... continue
                        continue

                    tokens = range(first_affected_token, last_affected_token + 1)

                    token_indices[k].extend(tokens)

                    if k != "template":
                        non_template_tokens.update(tokens)

            token_indices["template"] = [
                i for i in range(batch.length[statement_index]) if i not in non_template_tokens
            ]

            if scoring_masks is not None:
                i = 0
                remapped: list[int] = []
                for m in scoring_masks[statement_index]:
                    if m:
                        remapped.append(i)
                        i += 1
                    else:
                        remapped.append(-1)

                token_indices = {
                    k: [remapped[i] for i in v if scoring_masks[statement_index][i] if remapped[i] > 0]
                    for k, v in token_indices.items()
                }

            token_roles.append(token_indices)
        return token_roles

    @classmethod
    def from_model(cls, model: Union[str, PreTrainedModel], model_type: Optional[str] = None, **kw) -> "Evaluator":
        if cls is Evaluator:
            if model_type is None:
                if isinstance(model, str):
                    model_type = cls._infer_type_from_name(model)
                else:
                    model_type = cls._infer_type_from_object(model)

            evaluator_class: type[Evaluator]
            if model_type == "MLM":
                evaluator_class = MaskedLMEvaluator
            elif model_type == "CLM":
                evaluator_class = CausalLMEvaluator
            else:
                log.error("The class could not be instantiated.")
                msg = "The model is not compatible."
                raise ValueError(msg)

            return evaluator_class.from_model(model=model, model_type=model_type, **kw)
        else:
            return cast("Evaluator", super().from_model(model=model, model_type=model_type, **kw))

    @classmethod
    def _get_reduction_function(cls, reduction: str) -> Callable:
        if reduction == "sum":
            return sum
        elif reduction == "mean":
            return lambda x: sum(x) / len(x)
        elif reduction is None:
            return lambda x: x
        else:
            msg = f"Invalid reduction option '{reduction}'. \
                Choose either 'sum', 'mean' or None (for each token)."
            raise ValueError(msg)


class MaskedLMEvaluator(MaskedLMScorer, Evaluator):
    def get_result_metadata(self, **kw) -> dict[str, Any]:
        return {"pll_metric": self.pll_metric, **super().get_result_metadata(**kw)}

    def encode(
        self, statements: Sequence[str], span_roles: Sequence[SpanRoles]
    ) -> tuple[BatchEncoding, Sequence[ScoringMask]]:
        """Encode the statements using the tokenizer and create an appropriate scoring mask.

        In case the conditional scores need to be created, set the scoring mask accordingly.
        """
        batch = self.tokenizer(
            list(statements),
            return_tensors="pt",
            padding=True,
            return_special_tokens_mask=True,
            return_length=True,
            return_attention_mask=True,
        )

        if self.conditional_score:
            token_roles = self.derive_token_roles(batch, span_roles, scoring_masks=None)

            scoring_masks = []

            for input_ids, roles in zip(batch["input_ids"], token_roles):
                mask = [False] * len(input_ids)

                # Only set the mask to true where a token is part of the answer
                for i in roles["answer"]:
                    mask[i] = True
                scoring_masks.append(mask)

        else:
            scoring_masks = [(~mask.bool()).tolist() for mask in batch["special_tokens_mask"]]

        return batch, scoring_masks


class CausalLMEvaluator(CausalLMScorer, Evaluator):
    def encode(
        self, statements: Sequence[str], span_roles: Sequence[SpanRoles]
    ) -> tuple[BatchEncoding, Sequence[ScoringMask]]:
        """Encode the statements using the tokenizer and create an appropriate scoring mask.

        In case the conditional scores need to be created, set the scoring mask accordingly.
        """
        batch = self.tokenizer(
            list(statements),
            return_tensors="pt",
            padding=True,
            return_special_tokens_mask=True,
            return_length=True,
            return_attention_mask=True,
        )

        scoring_masks = self.default_scoring_mask(batch)

        if self.conditional_score:
            token_roles = self.derive_token_roles(batch, span_roles, scoring_masks=None)

            for i, roles in enumerate(token_roles):
                # In autoregressive models, we can exclude everything leading up to the first part of the answer
                answer_start = min(roles["answer"])
                scoring_masks[i][:answer_start] = [False] * answer_start

        return batch, scoring_masks
