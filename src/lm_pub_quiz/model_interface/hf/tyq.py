# ruff: noqa
# type: ignore
from lm_pub_quiz.model_interface.hf.util import derive_token_roles_internal
from functools import reduce
from tkinter import Text
import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Optional, Union

import torch
import torch.nn.functional as torch_func
from transformers import BatchEncoding

from lm_pub_quiz.model_interface.hf import HFModelInterface
from lm_pub_quiz.types import ItemScores, ItemTokenScoresAndRoles, TextRoles
from lm_pub_quiz.util import iter_batches

log = logging.getLogger(__name__)


class TyQModelInterface(HFModelInterface):
    _reduction_warned = False
    default_reduction = "tyq"

    @property
    def mask_token(self) -> int:
        """Return the mask token id used by the tokenizer."""

        token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        assert isinstance(token_id, int)
        return token_id

    def score_statement_options(
        self,
        statement_options: Iterable[Sequence[str]],
        *,
        text_roles: Optional[Iterable[Sequence[TextRoles]]] = None,
        **kw,
    ) -> Iterator[Union[ItemTokenScoresAndRoles, ItemScores]]:
        if text_roles is None:
            msg = "`text_roles` needs to be set for the tyq model interface."
            raise ValueError(msg)

        for options, roles in zip(statement_options, text_roles):
            batch = self.tokenizer(
                list(options),
                padding=True,
                return_tensors="pt",
                add_special_tokens=True,
                return_attention_mask=True,
            )

            token_roles = derive_token_roles_internal(batch=batch, text_roles=roles)

            reduced_batch: list[dict[str, Any]] = []
            num_tokens_map: dict[int, int] = {}

            reduced_batch_indeces: list[int] = []

            all_labels: list[torch.Tensor] = []

            for batch_index, roles in enumerate(token_roles):
                num = len(roles["answer"])

                labels: torch.Tensor = batch["input_ids"][batch_index, roles["answer"]].clone()

                # Mask all answer tokens
                input_ids: torch.Tensor = batch["input_ids"][batch_index].clone()
                input_ids[roles["answer"]] = self.mask_token

                if num not in num_tokens_map:
                    reduced_batch.append(
                        {
                            "input_ids": input_ids,
                            "masked_tokens": list(roles["answer"]),
                            "representative": batch_index,
                        }
                    )
                    num_tokens_map[num] = len(reduced_batch) - 1
                    reduced_batch_index = num_tokens_map[num]

                else:
                    reduced_batch_index = num_tokens_map[num]

                    assert (reduced_batch[reduced_batch_index]["input_ids"] == input_ids).all()
                    assert reduced_batch[reduced_batch_index]["masked_tokens"] == roles["answer"]

                all_labels.append(labels)
                reduced_batch_indeces.append(reduced_batch_index)

            # Process the model input
            representatives = [d["representative"] for d in reduced_batch]
            model_input = {k: v[representatives] for k, v in batch.items() if k != "input_ids"}
            model_input["input_ids"] = torch.stack([d["input_ids"] for d in reduced_batch])

            model_input = BatchEncoding(model_input).to(self.device)

            with torch.no_grad():
                model_output = self.model(**model_input)

            # Read out the scores
            batch_preds = torch.nn.functional.log_softmax(model_output.logits, -1)

            option_scores = []

            for batch_index, reduced_batch_index in enumerate(reduced_batch_indeces):
                scores = batch_preds[
                    reduced_batch_index, reduced_batch[reduced_batch_index]["masked_tokens"], all_labels[batch_index]
                ]

                log.debug("Option %d: %s", batch_index, str(scores))

                option_scores.append(scores.mean().item())

            log.debug(str(option_scores))

            yield option_scores
