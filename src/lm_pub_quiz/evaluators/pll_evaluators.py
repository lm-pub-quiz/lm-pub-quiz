"""MultipleChoiceEvaluator"""

import logging
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
    overload,
)

from tqdm.auto import tqdm
from transformers import (
    BatchEncoding,
    PreTrainedModel,
)

from lm_pub_quiz.evaluators.base import BaseEvaluator
from lm_pub_quiz.evaluators.scoring_mixins import CausalLMScoringMixin, MaskedLMScoringMixin, PLLScoringBaseMixin
from lm_pub_quiz.types import (
    EachTokenReturnFormat,
    ReducedReturnFormat,
    ScoredToken,
    ScoringMask,
    SpanRoles,
    TokenRoles,
)

tqdm.pandas()


log = logging.getLogger(__name__)


class Evaluator(BaseEvaluator, PLLScoringBaseMixin):
    """Base class for PLL-based evaluation classes.

    Use `Evaluator.from_model` to create a suitable model-specific `Evaluator` instance.
    """

    def __init__(
        self,
        *,
        conditional_score: bool = False,
        ensure_bos_token_added: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conditional_score = conditional_score

        self.ensure_bos_token_added = ensure_bos_token_added
        self._bos_token_warning_issued = False

    def get_result_metadata(self, **kw) -> dict[str, Any]:
        return {"ensure_bos_token_added": self.ensure_bos_token_added, **super().get_result_metadata(**kw)}

    @classmethod
    def from_model(cls, model: Union[str, PreTrainedModel], model_type: Optional[str] = None, **kw) -> "Evaluator":
        """Create an evaluator instance for the given model.

        In some cases, the model type can be derived from the model itself. To ensure
        the right type is chosen, it's recommended to set `model_type` manually.

        Parameters:
            model str | PreTrainedModel: The model to evaluate.
            model_type str | None: The type of model (determines the scoring scheme to be used).

        Returns:
            Evaluator: The evaluator instance suitable for the model.
        """
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
        """Calculate sequence scores using the Casual Language Model.

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

        token_roles_internal = self._derive_token_roles_internal(batch=batch, span_roles=span_roles)

        token_scores: list[list[float]] = self.score_statements(
            batch,
            scoring_masks=scoring_masks,
            batch_size=batch_size,
            token_roles_internal=token_roles_internal,
        )

        if reduction is None:
            scored_tokens: list[list[ScoredToken]]

            decoded_tokens = [
                [token for token, m in zip(tokens, mask) if m]
                for tokens, mask in zip(self.decode_tokens(batch), scoring_masks)
            ]

            scored_tokens = [
                list(zip(tokens, token_scores)) for tokens, token_scores in zip(decoded_tokens, token_scores)
            ]

            token_roles_output = self._remap_token_roles(
                token_roles_internal=token_roles_internal,
                scoring_masks=scoring_masks,
            )

            return list(zip(scored_tokens, token_roles_output))

        else:
            reduction_func = self._get_reduction_function(reduction)
            return [reduction_func(s) for s in token_scores]

    def decode_tokens(
        self, batch: BatchEncoding, scoring_masks: Optional[Sequence[ScoringMask]] = None
    ) -> list[list[str]]:
        if scoring_masks is None:
            return [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in batch["input_ids"]]
        else:
            return [batch.tokens(i)[scoring_mask] for i, scoring_mask in enumerate(scoring_masks)]

    def _derive_token_roles(
        self,
        *,
        batch: BatchEncoding,
        span_roles: Sequence[SpanRoles],
        scoring_masks: Optional[Sequence[ScoringMask]],
        output_indices: bool,
    ) -> Sequence[TokenRoles]:
        warnings.warn(
            "We recommennd using the more explicit functions `_derive_token_roles_internal` and `_remap_token_roles`.",
            stacklevel=2,
        )

        token_roles = self._derive_token_roles_internal(
            batch=batch,
            span_roles=span_roles,
        )

        if not output_indices:
            return token_roles
        elif scoring_masks is None:
            msg = "Cannot remap token roles if the scoring masks are not set."
            raise ValueError(msg)
        else:
            return self._remap_token_roles(token_roles_internal=token_roles, scoring_masks=scoring_masks)

    def _derive_token_roles_internal(
        self,
        batch: BatchEncoding,
        span_roles: Sequence[SpanRoles],
    ) -> Sequence[TokenRoles]:
        """Derive which tokens belong to the subject, answer, and template.

        If the scoring mask is given, the token indices refer to the resulting scores.
        """
        token_roles: list[TokenRoles] = []

        non_template_tokens: set[int]
        roles: dict[str, list[int]]

        for statement_index, spans in enumerate(span_roles):
            non_template_tokens = set()

            # For the statement, determine which tokens belong to each role
            roles = {k: [] for k in spans.keys()}

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

                    roles[k].extend(tokens)

                    if k != "template":
                        non_template_tokens.update(tokens)

            roles["template"] = [i for i in range(batch.length[statement_index]) if i not in non_template_tokens]
            token_roles.append(roles)

        return token_roles

    def _remap_token_roles(
        self, *, token_roles_internal: Sequence[TokenRoles], scoring_masks: Sequence[ScoringMask]
    ) -> Sequence[TokenRoles]:
        token_roles_output = []

        for statement_index, (roles, mask) in enumerate(zip(token_roles_internal, scoring_masks)):
            # Remap the token indices to the index of actually scored tokens
            # (for retrieval of token log-likihodds)
            i = 0
            remapped: list[int] = []
            for m in scoring_masks[statement_index]:
                if m:
                    remapped.append(i)
                    i += 1
                else:
                    remapped.append(-1)

            token_indices = {k: [remapped[i] for i in v if mask[i] if remapped[i] > 0] for k, v in roles.items()}

            token_roles_output.append(token_indices)
        return token_roles_output

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

    @abstractmethod
    def encode(
        self, statements: Sequence[str], span_roles: Sequence[SpanRoles]
    ) -> tuple[BatchEncoding, Sequence[ScoringMask]]:
        """Encode the statements using the tokenizer and create an appropriate scoring mask.

        In case the conditional scores need to be created, set the scoring mask accordingly.
        """


class MaskedLMEvaluator(MaskedLMScoringMixin, Evaluator):
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
            token_roles = self._derive_token_roles_internal(batch, span_roles)

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


class CausalLMEvaluator(CausalLMScoringMixin, Evaluator):
    @contextmanager
    def add_bos_token(self):
        _add_bos_token = self.tokenizer.add_bos_token
        self.tokenizer.add_bos_token = True
        yield
        self.tokenizer.add_bos_token = _add_bos_token

    def encode(
        self, statements: Sequence[str], span_roles: Sequence[SpanRoles]
    ) -> tuple[BatchEncoding, Sequence[ScoringMask]]:
        """Encode the statements using the tokenizer and create an appropriate scoring mask.

        In case the conditional scores need to be created, set the scoring mask accordingly.
        """
        with self.add_bos_token():
            batch = self.tokenizer(
                list(statements),
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
                return_special_tokens_mask=True,
                return_length=True,
                return_attention_mask=True,
            )
            if batch["input_ids"][0, 0] != self.tokenizer.bos_token_id:
                if self.ensure_bos_token_added:
                    # Issue a warning
                    if not self._bos_token_warning_issued:
                        self._bos_token_warning_issued = True
                        log.warning(
                            "The tokenizer did not add a BOS-token, even with `add_bos_token=True`. "
                            "A BOS token was added manually, to prevent this (and keep the default behavior of the "
                            "tokenizer), set `ensure_bos_token_added=False`."
                        )

                    # Add the bos token manually
                    statements = [self.tokenizer.bos_token + s for s in statements]
                    batch = self.tokenizer(
                        list(statements),
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=True,
                        return_special_tokens_mask=True,
                        return_length=True,
                        return_attention_mask=True,
                    )
                    batch["special_tokens_mask"][:, 0] = 1
                elif not self._bos_token_warning_issued:
                    self._bos_token_warning_issued = True
                    log.info(
                        "The tokenizer did not add a BOS-token, even with `add_bos_token=True`. "
                        "To add the BOS token manually in this case, set `ensure_bos_token_added=True`."
                    )

        scoring_masks = self.default_scoring_mask(batch)

        if self.conditional_score:
            token_roles = self._derive_token_roles_internal(batch, span_roles)

            for i, roles in enumerate(token_roles):
                # In autoregressive models, we can exclude everything leading up to the first part of the answer
                answer_start = min(roles["answer"])
                scoring_masks[i][:answer_start] = [False] * answer_start

        return batch, scoring_masks
