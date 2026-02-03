# ruff: noqa
# type: ignore
import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Optional, Union

import torch
import torch.nn.functional as torch_func

from lm_pub_quiz.model_interface.hf import HFModelInterface
from lm_pub_quiz.types import ItemScores, ItemTokenScoresAndRoles, TextRoles
from lm_pub_quiz.util import iter_batches

log = logging.getLogger(__name__)


class TyQModelInterface(HFModelInterface):
    _reduction_warned = False
    default_reduction = "tyq"

    def score_statement_options(
        self,
        *,
        statements: Iterable[Sequence[str]],
        roles: Optional[Iterable[Sequence[TextRoles]]] = None,
        **kw,
    ) -> Iterator[Union[ItemTokenScoresAndRoles, ItemScores]]:
        pass
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

        all_log_probs: list[torch.Tensor] = []

        # process the sentences using the model
        with torch.no_grad():
            for batch, masks in zip(
                iter_batches(encoded_sentences, batch_size=batch_size),
                iter_batches(masked_indices, batch_size=batch_size),
            ):
                logits = self.model(**batch).logits
                for logit, mask in zip(logits, masks):
                    log_probs = torch_func.log_softmax(logit[mask], dim=-1).to(self.device)
                    all_log_probs.append(log_probs)

        # read out the mean log probs
        scores: list[float] = []
        for length, token_ids in zip(encoded_answers.length, encoded_answers.input_ids):
            sentence_id: int = required_lens.index(length)

            answer_score = all_log_probs[sentence_id][:, token_ids].mean().item()
            scores.append(answer_score)

        # return the resulting scores
        return scores
