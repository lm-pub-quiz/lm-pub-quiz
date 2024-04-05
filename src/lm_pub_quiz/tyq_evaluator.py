import logging
from typing import Iterable, List, Optional, Union, cast

import torch
import torch.nn.functional as torch_func
from transformers import (
    AutoModelForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from lm_pub_quiz.evaluator import BaseEvaluator, EachTokenReturnFormat, ReducedReturnFormat

log = logging.getLogger(__name__)


class TyQEvaluator(BaseEvaluator):
    _reduction_warned = False
    default_reduction = "tyq"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        tokenizer: Union[str, PreTrainedTokenizerFast, None] = None,
        *,
        device: Union[str, int] = "cpu",
        capitalize: bool = True,
        **kwargs,
    ):
        super().__init__(model, device=device, tokenizer=tokenizer)
        self.capitalize = capitalize

        if isinstance(model, str):
            self.model = AutoModelForMaskedLM.from_pretrained(model, return_dict=True, **kwargs)
            if self.device == "auto":
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model, device_map=self.device, return_dict=True, **kwargs
                )
            # self.model.to(self.device)
        else:
            self.model = model

        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()

    def evaluate_instance(
        self,
        *,
        template: str,
        answers: Iterable[str],
        reduction: Optional[str] = "tyq",
        batch_size: int = -1,  # noqa: ARG002
        subject: Optional[str] = None,
        print_ranking: bool = False,
    ) -> Union[ReducedReturnFormat, EachTokenReturnFormat]:
        if "[Y]" not in template:
            msg = 'Provided sentence is missing a placeholder ("[Y]") used for answers.'
            raise ValueError(msg)

        if reduction != "tyq" and not self._reduction_warned:
            log.warning("Reduction other than 'tyq' are being ignored ('%s' set).", str(reduction))

        results = cast(
            ReducedReturnFormat,
            self.score_answers(
                template=template,
                answers=answers,
                subject=subject,
            ),
        )

        if print_ranking:
            self.print_ranking(answers, results)

        return results

    def score_answers(
        self, *, template: str, answers: Iterable[str], subject: Optional[str]
    ) -> Union[ReducedReturnFormat, EachTokenReturnFormat]:
        # tokenize the answers
        encoded_answers = self.tokenizer(list(answers), return_length=True, padding=False)

        required_lens = list(set(encoded_answers.length))

        # prepare the masked sentences
        sentences = [
            self.fill_template(
                template=template, subject=subject, answer=" ".join([self.tokenizer.mask_token] * num_masks)
            )
            for num_masks in required_lens
        ]

        encoded_sentences = self.tokenizer(sentences, return_tensors="pt", padding=True)
        encoded_sentences.to(self.device)

        all_masked_indices = [
            [i for i, t in enumerate(input_ids) if t == self.tokenizer.mask_token_id]
            for input_ids in encoded_sentences.input_ids
        ]

        assert all(len(masked) == length for length, masked in zip(required_lens, all_masked_indices))

        # process the sentences using the model
        with torch.no_grad():
            logits = self.model(**encoded_sentences).logits
            all_log_probs = torch_func.log_softmax(logits, dim=-1).cpu()

        # read out the mean log probs
        scores: List[float] = []

        for length, token_ids in zip(encoded_answers.length, encoded_answers.input_ids):
            sentence_id: int = required_lens.index(length)
            masked_indices = all_masked_indices[sentence_id]

            assert length == len(masked_indices)

            answer_score = all_log_probs[sentence_id, masked_indices, token_ids].mean().item()
            scores.append(answer_score)

        # return the resulting scores
        return scores

    @classmethod
    def from_model(
        cls, model: Union[str, PreTrainedModel], *, model_type: Optional[str] = None, **kwargs
    ) -> "TyQEvaluator":
        model_str: str = cls._get_model_name(model)
        if model_type is None:
            log.debug("Inferring model type for %s.", model_str)
            model_type = cls._infer_type_from_name(model_str)
            log.debug("Inferred model type: %s", model_type)

        assert model_type == "MLM"

        return cls(model, **kwargs)
