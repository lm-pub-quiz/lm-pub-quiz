import itertools
import logging
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
from tqdm.auto import tqdm
from transformers import (
    BatchEncoding,
)

from lm_pub_quiz.model_interface.base import PLLModelInterfaceMixin
from lm_pub_quiz.model_interface.hf.base import HFModelInterface
from lm_pub_quiz.model_interface.hf.util import derive_token_roles_internal, get_reduction_function, remap_token_roles
from lm_pub_quiz.types import (
    ScoredToken,
    ScoringMask,
    StatementScore,
    TextRoles,
    TokenScoresAndRoles,
)
from lm_pub_quiz.util import iter_batches

tqdm.pandas()


log = logging.getLogger(__name__)


class CLMInterface(PLLModelInterfaceMixin, HFModelInterface):
    def __init__(self, *args, ensure_bos_token_added: bool = True, conditional_score: bool = False, **kw):
        super().__init__(*args, **kw)

        self.ensure_bos_token_added: bool = ensure_bos_token_added
        self.conditional_score: bool = conditional_score
        self._bos_token_warning_issued: bool = False

    def get_metadata(self) -> dict[str, Any]:
        return {
            "ensure_bos_token_added": self.ensure_bos_token_added,
            "conditional_score": self.conditional_score,
            **super().get_metadata(),
        }

    def default_scoring_mask(self, batched_statements: BatchEncoding) -> Sequence[ScoringMask]:
        return [(~mask.bool()).tolist() for mask in batched_statements["special_tokens_mask"]]

    @contextmanager
    def add_bos_token(self):
        _add_bos_token = self.tokenizer.add_bos_token
        self.tokenizer.add_bos_token = True
        yield
        self.tokenizer.add_bos_token = _add_bos_token

    def encode(
        self, statements: Sequence[str], roles: Optional[Sequence[TextRoles]]
    ) -> tuple[BatchEncoding, Sequence[ScoringMask]]:
        """Encode the statements using the tokenizer and create an appropriate scoring mask.

        In case the conditional scores need to be created, set the scoring mask accordingly.
        """
        with self.add_bos_token():
            batch: BatchEncoding = self.tokenizer(
                statements,
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
            if roles is None:
                msg = "When using conditional scoring, roles (of the text segments) need to be passed."
                raise ValueError(msg)

            token_roles = derive_token_roles_internal(batch=list(batch), text_roles=list(roles))

            for i, tr in enumerate(token_roles):
                # In autoregressive models, we can exclude everything leading up to the first part of the answer
                answer_start = min(tr["answer"])
                scoring_masks[i][:answer_start] = [False] * answer_start

        return batch, scoring_masks

    def decode_tokens(
        self, batch: BatchEncoding, scoring_masks: Optional[Sequence[ScoringMask]] = None
    ) -> list[list[str]]:
        if scoring_masks is None:
            return [self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in batch["input_ids"]]
        else:
            return [batch.tokens(i)[scoring_mask] for i, scoring_mask in enumerate(scoring_masks)]

    def score_statements(
        self,
        statements: Iterable[str],
        *,
        text_roles: Optional[Iterable[TextRoles]] = None,
        batch_size: Optional[int] = None,
        **kw,
    ) -> Union[Iterable[TokenScoresAndRoles], Iterable[StatementScore]]:
        """Score individual texts (independent of the other options) using the Casual Language Model.

        Parameters:
            statements: The statements to score.
            text_roles: Which parts of the statement are the answer, template, and subject.

        Returns:
            Scores (or scores and roles) per statement
        """

        reduction = kw.get("reduction", self.default_reduction)
        batch_size = batch_size or self.batch_size

        batch: BatchEncoding
        scoring_masks: Sequence[ScoringMask]

        if text_roles is None:
            batches = zip(iter_batches(statements, batch_size=batch_size), itertools.repeat(None))
        else:
            batches = zip(
                iter_batches(statements, batch_size=batch_size), iter_batches(text_roles, batch_size=batch_size)
            )

        for statement_batch, role_batch in batches:
            if role_batch is not None:
                role_batch_list = list(role_batch)
            else:
                role_batch_list = None

            batch, scoring_masks = self.encode(
                statements=statement_batch,
                roles=role_batch_list,
            )

            if role_batch_list is not None:
                token_roles_internal = derive_token_roles_internal(batch=batch, text_roles=list(role_batch_list))

            scoring_masks = self.default_scoring_mask(batch)

            # Forward the batch through the model
            with torch.no_grad():
                batch.pop("special_tokens_mask")
                batch.pop("length")
                model_output = self.model(**batch)

            # Shift so that tokens < n predict n
            batch_logits = model_output.logits[:, :-1]
            batch_labels = batch["input_ids"][:, 1:]

            batch_token_scores: list[list[float]] = []

            for i, mask in enumerate(scoring_masks):
                if mask is None:
                    logits = batch_logits[i, :].contiguous()
                    labels = batch_labels[i, :].contiguous()
                else:
                    logits = batch_logits[i, mask[1:]].contiguous()
                    labels = batch_labels[i, mask[1:]].contiguous()

                preds = torch.nn.functional.log_softmax(logits, -1)

                batch_token_scores.append(preds[torch.arange(labels.size(0)), labels].cpu().tolist())

            if reduction is None:
                scored_tokens: list[list[ScoredToken]]

                decoded_tokens = [
                    [token for token, m in zip(tokens, mask) if m]
                    for tokens, mask in zip(self.decode_tokens(batch), scoring_masks)
                ]

                scored_tokens = [
                    list(zip(tokens, token_scores)) for tokens, token_scores in zip(decoded_tokens, batch_token_scores)
                ]

                if role_batch is not None:
                    token_roles_output = [
                        remap_token_roles(
                            token_roles_internal=roles,
                            scoring_mask=mask,
                        )
                        for mask, roles in zip(scoring_masks, token_roles_internal)
                    ]
                else:
                    token_roles_output = [{} for _ in scoring_masks]

                yield from zip(scored_tokens, token_roles_output)

            else:
                reduction_func = get_reduction_function(reduction)
                yield from map(reduction_func, batch_token_scores)
