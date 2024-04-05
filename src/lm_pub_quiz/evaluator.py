"""MultipleChoiceEvaluator"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

import pandas as pd
from minicons import scorer
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from lm_pub_quiz.data import Dataset, DatasetResults, Relation, RelationResult
from lm_pub_quiz.metrics import RelationMetric
from lm_pub_quiz.metrics.base import MetricSpecification
from lm_pub_quiz.templating import Templater
from lm_pub_quiz.util import EachTokenReturnFormat, ReducedReturnFormat, parse_dumped_raw_results

tqdm.pandas()

log = logging.getLogger(__name__)


MultiMetricSpecification = Union[MetricSpecification, Sequence[MetricSpecification]]


class BaseEvaluator(ABC):
    default_reduction = "sum"

    _mlm_keywords: Tuple[str, ...] = ("bert",)
    _clm_keywords: Tuple[str, ...] = ("opt", "gpt", "llama", "bloom", "google/gemma", "mistral")

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        tokenizer: Union[str, PreTrainedTokenizerFast, None] = None,
        *,
        device: Union[str, int] = "cpu",
        subject_placeholder="[X]",
        answer_placeholder="[Y]",
    ):
        self._set_device(device)
        self.model_name: str = self._get_model_name(model)
        self.tokenizer = self._get_tokenizer(model, tokenizer)

        self.templater = Templater(subject_placeholder=subject_placeholder, answer_placeholder=answer_placeholder)

    @abstractmethod
    def evaluate_instance(
        self,
        *,
        template: str,
        answers: Iterable[str],
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

        relation_type = (
            "multiple instances per answer" if instances.duplicated("obj_id").any() else "single instance per answer"
        )

        relation_result = RelationResult(
            relation_code=relation.relation_code,
            instance_table=None,
            answer_space=relation.answer_space.copy(),
            metadata={
                "templates": relation.templates,
                "template_index": template_index,
                "model_name_or_path": self.model_name,
                "num_original_instances": len(relation),
                "subsampled": subsample,
                "time_start": datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                "relation_type": relation_type,
            },
        )

        template = relation.templates[template_index]

        evaluated_instances: List[Dict] = []

        metrics: List[RelationMetric] = []
        if metric is not None:
            if isinstance(metric, (str, RelationMetric)) or (
                isinstance(metric, type) and issubclass(metric, RelationMetric)
            ):
                metric = (metric,)

            for m in metric:
                m_obj: RelationMetric = RelationMetric.create_metric(m)
                m_obj.reset()
                metrics.append(m_obj)

        for _, r in tqdm(instances.iterrows()):
            row = r.to_dict()
            row["answer_idx"] = relation.answer_space.index.get_loc(row["obj_id"])

            pll_scores = self.evaluate_instance(
                template=template,
                answers=relation.answer_space,
                batch_size=batch_size,
                subject=str(row["sub_label"]),
                reduction=reduction,
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

        if create_instance_table:
            relation_result._instance_table = pd.DataFrame(evaluated_instances, index=instances.index)

        for m in metrics:
            relation_result.metric_values.update(m.compute())

        relation_result.metadata["time_end"] = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

        self.update_result_metadata(relation_result, reduction=reduction)

        return relation_result

    def evaluate_dataset(
        self,
        dataset: Dataset,
        template_index: int = 0,
        *,
        batch_size: int = 1,
        subsample: Optional[int] = None,
        save_path: Optional[Path] = None,
        reduction: Optional[str] = "default",
        create_instance_table: bool = True,
        metric: Optional[MultiMetricSpecification] = None,
    ) -> DatasetResults:
        """Evaluate the model on all relations in the dataset."""
        if reduction == "default":
            reduction = self.default_reduction

        dataset_results = DatasetResults()

        log.debug("Evaluating `%s` on `%s`", self.model_name, dataset.name)

        for relation in dataset:
            try:
                log.info("Evaluating `%s` on %s.", self.model_name, relation)
                relation_result = self.evaluate_relation(
                    relation,
                    template_index=template_index,
                    batch_size=batch_size,
                    subsample=subsample,
                    reduction=reduction,
                    create_instance_table=create_instance_table,
                    metric=metric,
                )
                self.update_result_metadata(relation_result, dataset=dataset)

                if save_path is not None:
                    relation_result._lazy_load_path = relation_result.save(save_path)
                    relation_result._instance_table = None

                dataset_results.append(relation_result)

            except RuntimeError:
                continue

        return dataset_results

    def update_result_metadata(self, result, **kw) -> None:
        if "dataset" in kw:
            dataset = kw["dataset"]
            result.metadata["dataset_path"] = dataset.path
            result.metadata["dataset_name"] = dataset.name

        if "reduction" in kw:
            result.metadata["reduction"] = kw["reduction"]

    def _set_device(self, device_input: Union[int, str]):
        if isinstance(device_input, int):
            self.device = f"cuda:{device_input}"
        else:
            self.device = device_input

        log.info("Device %s selected.", self.device)

    @staticmethod
    def _get_model_name(model: Union[str, PreTrainedModel]) -> str:
        if isinstance(model, str):
            return model
        else:
            return model.base_model_prefix

    @staticmethod
    def print_ranking(answers: Iterable[str], scores: List[float]) -> None:
        data = zip(answers, scores)
        sorted_data = sorted(data, key=lambda x: x[1], reverse=False)
        max_str_length = max([len(item[0]) for item in sorted_data])

        # Print header
        print(f"{'Rank':<5}{'Word':<{max_str_length + 2}}{'Score':<10}")  # noqa: T201
        print("-" * (max_str_length + 26))  # noqa: T201

        # Print each item
        for rank, (word, score) in enumerate(sorted_data, 1):
            print(f"{rank:<5}{word:<{max_str_length + 2}}{score:<10}")  # noqa: T201

    @staticmethod
    def _get_tokenizer(
        model: Union[str, PreTrainedModel], tokenizer: Union[str, PreTrainedTokenizerFast, None]
    ) -> PreTrainedTokenizerFast:
        """Retrieve a tokenizer that matches the model or tokenizer string."""
        if tokenizer is None:
            if isinstance(model, str):
                return AutoTokenizer.from_pretrained(model, use_fast=True)
            else:
                return AutoTokenizer.from_pretrained(model.config.name_or_path, use_fast=True)
        elif isinstance(tokenizer, str):
            return AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        else:
            return tokenizer

    @staticmethod
    def _chunks(sentences: Iterable[str], batch_size: int) -> Iterator[List[str]]:
        """Yield successive n-sized chunks from list."""

        if batch_size > 0:
            it = iter(sentences)

            while batch := list(islice(it, batch_size)):
                yield batch
        else:
            yield list(sentences)

    @classmethod
    def _infer_type_from_name(cls, model_name_or_path: str) -> Literal["MLM", "CLM"]:
        """Infer the type of model (MLM or CLM) based on the model name."""
        if any(k in model_name_or_path.lower() for k in cls._mlm_keywords):
            return "MLM"
        elif any(k in model_name_or_path.lower() for k in cls._clm_keywords):
            return "CLM"
        else:
            msg = f"Cannot infer model type from the `model_name_or_path`: '{model_name_or_path}'."
            log.error(msg)
            raise ValueError(msg)

    # Return only the textual statement
    @overload
    def fill_template(
        self,
        template: str,
        answer: str,
        *,
        subject: Optional[str] = None,
        capitalize: bool = True,
        return_token_indices: Literal[False] = False,
        return_prefix: Literal[False] = False,
        return_suffix: Literal[False] = False,
        include_special_tokens: bool = True,
    ) -> str: ...

    # Return the token indices
    @overload
    def fill_template(
        self,
        template: str,
        answer: str,
        *,
        subject: Optional[str] = None,
        capitalize: bool = True,
        return_token_indices: Literal[True],
        return_prefix: Literal[False] = False,
        return_suffix: Literal[False] = False,
        include_special_tokens: bool = True,
    ) -> Tuple[str, Dict[str, List[int]]]: ...

    # Return the prefix and the stimulus
    @overload
    def fill_template(
        self,
        template: str,
        answer: str,
        *,
        subject: Optional[str] = None,
        capitalize: bool = True,
        return_token_indices: Literal[False] = False,
        return_prefix: Literal[True],
        return_suffix: Literal[False] = False,
        include_special_tokens: bool = True,
    ) -> Tuple[str, str]: ...

    # Return the prefix, stimulus, and suffix
    @overload
    def fill_template(
        self,
        template: str,
        answer: str,
        *,
        subject: Optional[str] = None,
        capitalize: bool = True,
        return_token_indices: Literal[False] = False,
        return_prefix: Literal[True],
        return_suffix: Literal[True],
        include_special_tokens: bool = True,
    ) -> Tuple[str, str, str]: ...

    def fill_template(
        self,
        template: str,
        answer: str,
        *,
        subject: Optional[str] = None,
        capitalize: bool = True,
        return_token_indices: bool = False,
        return_prefix: bool = False,
        return_suffix: bool = False,
        include_special_tokens: bool = True,
    ) -> Union[str, Tuple[str, Dict[str, List[int]]], Tuple[str, str], Tuple[str, str, str]]:
        """Create a sentence/text based on a template and an anwser.

        Parameters:
            template (str): The template to use. Should contain an answer placeholder.
            answer (str): The answer to fill in.
            subject (str or None): The subject to fill in (or ignore if None).
            return_prefix (bool): If set to true, returns the prefix (for conditional score computation).
            return_suffix (bool): If set to true, returns the suffix (for conditional score computation).
        """
        if return_token_indices and return_prefix:
            msg = "Indics can only be returned if return_prefix is not set."
            raise ValueError(msg)

        if not return_prefix and return_suffix:
            msg = "Sufffix can only be used if prefix is also set."
            raise ValueError(msg)

        text, spans = self.templater.replace_placeholders(
            template=template, subject=subject, answer=answer, capitalize=capitalize
        )

        if not return_token_indices and not return_prefix:
            return text

        elif return_token_indices:
            _, token_indices = self.templater.tokenize_with_span_dict(
                tokenizer=self.tokenizer,
                text=text,
                spans=spans,
                include_template_indices=True,
                include_special_tokens=include_special_tokens,
            )

            return text, token_indices

        else:
            prefix_end = min(start for start, _ in spans["answer"])

            if not return_suffix:
                return text[:prefix_end], text[prefix_end:]

            else:
                suffix_start = max(end for _, end in spans["answer"])
                return text[:prefix_end], text[prefix_end:suffix_start], text[suffix_start:]


class Evaluator(BaseEvaluator):
    """Perplexity-based evaluator base class."""

    def evaluate_instance(
        self,
        *,
        template: str,
        answers: Iterable[str],
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

        results: List = []
        for answer in self._chunks(answers, batch_size):
            results += self.score_answers(
                template=template,
                answers=answer,
                reduction=reduction,
                subject=subject,
            )

        if print_ranking:
            self.print_ranking(answers, results)

        return results

    @overload
    @abstractmethod
    def score_answers(
        self, *, template: str, answers: List[str], reduction: None, subject: Optional[str] = None
    ) -> EachTokenReturnFormat: ...

    @overload
    @abstractmethod
    def score_answers(
        self, *, template: str, answers: List[str], reduction: str, subject: Optional[str] = None
    ) -> ReducedReturnFormat: ...

    @abstractmethod
    def score_answers(
        self, *, template: str, answers: List[str], reduction: Optional[str], subject: Optional[str] = None
    ) -> Union[ReducedReturnFormat, EachTokenReturnFormat]:
        """Score an answer given a template.

        This function must be implemented by child-classes for each model-type.
        """

    @classmethod
    def from_model(
        cls, model: Union[str, PreTrainedModel], *, model_type: Optional[str] = None, **kwargs
    ) -> "Evaluator":
        model_str: str = cls._get_model_name(model)

        if model_type is None:
            model_type = cls._infer_type_from_name(model_str)
            log.debug("Inferred type of model `%s`: %s", model_str, model_type)

        evaluator_class: Type["Evaluator"]
        if model_type == "MLM":
            evaluator_class = MaskedLMEvaluator
        elif model_type == "CLM":
            evaluator_class = CausalLMEvaluator
        else:
            log.error("The class could not be instantiated.")
            msg = "The model is not compatible."
            raise ValueError(msg)

        return evaluator_class(model, **kwargs)

    @classmethod
    def _get_reduction_function(cls, reduction: str) -> Callable:
        if reduction == "sum":
            return lambda x: x.sum(0)
        elif reduction == "mean":
            return lambda x: x.mean(0)
        elif reduction is None:
            return lambda x: x
        else:
            msg = f"Invalid reduction option '{reduction}'. \
                Choose either 'sum', 'mean' or None (for each token)."
            raise ValueError(msg)


class MaskedLMEvaluator(Evaluator):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        tokenizer: Union[str, PreTrainedTokenizerFast, None] = None,
        *,
        device: Union[str, int] = "cpu",
        conditional_score: bool = False,
        pll_metric: str = "within_word_l2r",
        capitalize: bool = True,
        **kwargs,
    ):
        super().__init__(model, device=device, tokenizer=tokenizer)
        self.scorer = scorer.MaskedLMScorer(model, device=self.device, tokenizer=self.tokenizer, **kwargs)

        self.pll_metric = pll_metric
        self.conditional_score = conditional_score
        self.capitalize = capitalize

    def fill_template(
        self,
        *args,
        include_special_tokens: bool = False,
        **kw,
    ):
        return super().fill_template(*args, include_special_tokens=include_special_tokens, **kw)

    @overload
    def score_answers(
        self, *, template: str, answers: List[str], reduction: None, subject: Optional[str] = None
    ) -> EachTokenReturnFormat: ...

    @overload
    def score_answers(
        self, *, template: str, answers: List[str], reduction: str, subject: Optional[str] = None
    ) -> ReducedReturnFormat: ...

    def score_answers(
        self, *, template: str, answers: List[str], reduction: Optional[str], subject: Optional[str] = None
    ) -> Union[ReducedReturnFormat, EachTokenReturnFormat]:
        """Calculates sequence scores using the Masked Language Model.

        Parameters:
            template str: The template to use (should contain a `[Y]` marker).
            answers List[str]: List of answers to calculate score for.

        Returns:
            List[float]: List of suprisals scores per sequence
        """
        probe_sentences: Sequence[str]
        if reduction is None:
            probe_sentences, token_indices = zip(
                *(
                    self.fill_template(
                        template, a, subject=subject, return_token_indices=True, capitalize=self.capitalize
                    )
                    for a in answers
                )
            )

            scores = self.scorer.token_score(probe_sentences, PLL_metric=self.pll_metric, surprisal=False)
            return list(zip(scores, token_indices))

        reduction_func: Callable = self._get_reduction_function(reduction)

        if not self.conditional_score:
            statements = [self.fill_template(template, a, subject=subject, capitalize=self.capitalize) for a in answers]
            return [
                score.item()
                for score in self.scorer.sequence_score(
                    statements, reduction=reduction_func, PLL_metric=self.pll_metric
                )
            ]
        else:
            prefixes, stimuli, suffixes = zip(
                *(
                    self.fill_template(
                        template,
                        answer,
                        subject=subject,
                        return_prefix=True,
                        return_suffix=True,
                        capitalize=self.capitalize,
                    )
                    for answer in answers
                )
            )

            return [
                score.item()
                for score in self.scorer.conditional_score(
                    prefix=list(prefixes),
                    stimuli=list(stimuli),
                    suffix=list(suffixes),
                    reduction=reduction_func,
                    PLL_metric=self.pll_metric,
                )
            ]

    def update_result_metadata(self, result, **kw) -> None:
        if "pll_metric" in kw:
            result.metadata["pll_metric"] = kw.pop("pll_metric")

        super().update_result_metadata(result, **kw)


class CausalLMEvaluator(Evaluator):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        tokenizer: Union[str, PreTrainedTokenizerFast, None] = None,
        *,
        device: Union[str, int] = "cpu",
        conditional_score: bool = False,
        capitalize: bool = True,
        **kwargs,
    ):
        super().__init__(model, tokenizer=tokenizer, device=device)
        self.scorer = scorer.IncrementalLMScorer(model, device=self.device, tokenizer=self.tokenizer, **kwargs)
        self.conditional_score = conditional_score
        self.capitalize = capitalize

    @overload
    def score_answers(
        self, *, template: str, answers: List[str], reduction: None, subject: Optional[str] = None
    ) -> EachTokenReturnFormat: ...

    @overload
    def score_answers(
        self, *, template: str, answers: List[str], reduction: str, subject: Optional[str] = None
    ) -> ReducedReturnFormat: ...

    def score_answers(
        self, *, template: str, answers: List[str], reduction: Optional[str], subject: Optional[str] = None
    ) -> Union[EachTokenReturnFormat, ReducedReturnFormat]:
        """Calculates sequence scores using the Casual Language Model.

        Parameters:
            template str: The template to use (should contain a `[Y]` marker).
            answers List[str]: List of answers to calculate score for.

        Returns:
            List[float]: List of suprisals scores per sequence
        """
        probe_sentences: Sequence[str]

        if reduction is None:
            probe_sentences, indices = zip(
                *(
                    self.fill_template(
                        template, a, subject=subject, return_token_indices=True, capitalize=self.capitalize
                    )
                    for a in answers
                )
            )

            scores = self.scorer.token_score(probe_sentences, surprisal=False)
            return list(zip(scores, indices))

        reduction_func = self._get_reduction_function(reduction)

        if not self.conditional_score:
            probe_sentences = [
                self.fill_template(template, a, subject=subject, capitalize=self.capitalize) for a in answers
            ]
            return [score.item() for score in self.scorer.sequence_score(probe_sentences, reduction=reduction_func)]
        else:
            prefixes, stimuli = zip(
                *(
                    self.fill_template(
                        template,
                        answer,
                        subject=subject,
                        capitalize=self.capitalize,
                        return_prefix=True,
                        return_suffix=False,
                    )
                    for answer in answers
                )
            )

            return [
                score.item()
                for score in self.scorer.conditional_score(
                    prefix=list(prefixes),
                    stimuli=list(stimuli),
                    reduction=reduction_func,
                )
            ]
