import logging
from typing import Iterable, List, Optional, Union, cast

import torch
import torch.nn.functional as torch_func

from lm_pub_quiz.evaluators.base import BaseEvaluator, EachTokenReturnFormat, ReducedReturnFormat
from lm_pub_quiz.evaluators.util import iter_batches

log = logging.getLogger(__name__)


class TyQEvaluator(BaseEvaluator):
    _reduction_warned = False
    default_reduction = "tyq"

    def evaluate_instance(
        self,
        *,
        template: str,
        answers: Iterable[str],
        reduction: Optional[str] = "tyq",
        batch_size: int = -1,
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
                batch_size=batch_size,
            ),
        )

        if print_ranking:
            self.print_ranking(answers, results)

        return results

    def score_answers(
        self, *, template: str, answers: Iterable[str], subject: Optional[str], batch_size: int = 1
    ) -> Union[ReducedReturnFormat, EachTokenReturnFormat]:
        # tokenize the answers
        encoded_answers = self.tokenizer(list(answers), return_length=True, padding=False)

        required_lens = list(set(encoded_answers.length))

        # prepare the masked sentences
        sentences = [
            self.replace_placeholders(
                template=template, subject=subject, answer=" ".join([self.tokenizer.mask_token] * num_masks)
            )[0]
            for num_masks in required_lens
        ]

        encoded_sentences = self.tokenizer(sentences, return_tensors="pt", padding=True)
        encoded_sentences.to(self.device)

        masked_indices = [
            [i for i, t in enumerate(input_ids) if t == self.tokenizer.mask_token_id]
            for input_ids in encoded_sentences.input_ids
        ]

        assert all(len(masked) == length for length, masked in zip(required_lens, masked_indices))

        all_log_probs: List[torch.Tensor] = []

        # process the sentences using the model
        with torch.no_grad():
            for batch, masks in zip(
                iter_batches(encoded_sentences, batch_size=batch_size),
                iter_batches(masked_indices, batch_size=batch_size),
            ):
                logits = self.model(**batch).logits
                for logit, mask in zip(logits, masks):
                    log_probs = torch_func.log_softmax(logit[mask], dim=-1).cpu()
                    all_log_probs.append(log_probs)

        # read out the mean log probs
        scores: List[float] = []
        for length, token_ids in zip(encoded_answers.length, encoded_answers.input_ids):
            sentence_id: int = required_lens.index(length)

            answer_score = all_log_probs[sentence_id][:, token_ids].mean().item()
            scores.append(answer_score)

        # return the resulting scores
        return scores
