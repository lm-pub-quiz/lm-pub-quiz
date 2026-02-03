import logging
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Optional,
    Union,
)

import torch
from tqdm.auto import tqdm
from transformers import (
    BatchEncoding,
)

from lm_pub_quiz.model_interface.base import PLLModelInterfaceMixin
from lm_pub_quiz.model_interface.hf.base import HFModelInterface
from lm_pub_quiz.model_interface.hf.util import derive_token_roles_internal, get_reduction_function, remap_token_roles
from lm_pub_quiz.types import ScoredToken, ScoringMask, StatementScore, TextRoles, TokenScoresAndRoles
from lm_pub_quiz.util import iter_batches

tqdm.pandas()


log = logging.getLogger(__name__)


class MLMInterface(PLLModelInterfaceMixin, HFModelInterface):
    def __init__(
        self,
        *args,
        pll_metric: str = "within_word_l2r",
        batch_size: int = 1,
        conditional_score: bool = False,
        preprocessing_batch_size: int = 1000,
        **kw,
    ) -> None:
        if pll_metric not in ("original", "within_word_l2r", "sentence_l2r", "answer_l2r+word_l2r"):
            msg = f"PLL strategy {pll_metric} not know."
            raise KeyError(msg)

        super().__init__(*args, **kw)

        self.pll_metric = pll_metric
        self.batch_size = batch_size
        self.conditional_score = conditional_score
        self.preprocessing_batch_size = preprocessing_batch_size

    def get_result_metadata(self) -> dict[str, Any]:
        return {"pll_metric": self.pll_metric, **super().get_result_metadata()}

    @property
    def mask_token(self) -> int:
        """Return the mask token id used by the tokenizer."""

        token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        assert isinstance(token_id, int)
        return token_id

    def score_statements(
        self,
        *,
        statements: Iterable[str],
        text_roles: Optional[Iterable[TextRoles]] = None,
        batch_size: Optional[int] = None,
        **kw,
    ) -> Union[Iterable[TokenScoresAndRoles], Iterable[StatementScore]]:
        """Calculate sequence scores using the Casual/Masked Language Model.

        Parameters:
            template str: The template to use (should contain a `[Y]` marker).
            answers list[str]: List of answers to calculate score for.

        Returns:
            list[float]: List of suprisals scores per sequence
        """
        if text_roles is None and self.conditional_score:
            msg = "`roles` need to be set to use `conditional_score`."
            raise ValueError(msg)

        # Tokenize, translate the text roles to token roles, and create scoring mask
        if text_roles is None:
            preprocessed_statements = (
                self.preprocess_statements(batch_statements)
                for batch_statements in iter_batches(statements, batch_size=self.preprocessing_batch_size)
            )
        else:
            preprocessed_statements = (
                self.preprocess_statements(batch_statements, text_roles=batch_text_roles)
                for batch_statements, batch_text_roles in zip(
                    iter_batches(statements, batch_size=self.preprocessing_batch_size),
                    iter_batches(text_roles, batch_size=self.preprocessing_batch_size),
                )
            )

        # Prepare repeated model calls for each statement
        masked_requests = (self.create_masked_requests(batch) for batch in preprocessed_statements)

        reduction = kw.get("reduction", self.default_reduction)
        if reduction is not None:
            reduction_func = get_reduction_function(reduction)

        # Process the extended requests
        for large_batch in masked_requests:
            statement_index: int = 0
            scored_tokens: list[ScoredToken] = []

            new_statement_index: int
            new_scored_token: ScoredToken

            for new_statement_index, new_scored_token in self.process_extended_statements(
                large_batch, batch_size=batch_size
            ):
                if statement_index != new_statement_index:
                    assert new_statement_index == statement_index + 1

                    ### Yield the return values ###
                    if reduction is None:
                        token_roles_output = remap_token_roles(
                            token_roles_internal=large_batch["token_roles_internal"][statement_index],
                            scoring_mask=large_batch["scoring_masks"][statement_index],
                        )

                        yield scored_tokens, token_roles_output

                    else:
                        reduction_func = get_reduction_function(reduction)
                        yield reduction_func([score for token, score in scored_tokens])

                    # Reset
                    statement_token_scores = []
                    statement_index = new_statement_index

                statement_token_scores.append(new_scored_token)

    def preprocess_statements(
        self, statements: Sequence[str], *, text_roles: Optional[Sequence[TextRoles]] = None
    ) -> BatchEncoding:
        """Tokenize statements, translate text roles (char level) to token roles and determine which tokens to score."""

        if text_roles is None and self.conditional_score:
            msg = "`roles` need to be set to use `conditional_score`."
            raise ValueError(msg)

        batch = self.tokenizer(
            list(statements),
            padding=False,
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )

        if text_roles is not None:
            batch["token_roles_internal"] = derive_token_roles_internal(batch=batch, text_roles=text_roles)

        batch["scoring_masks"]: list[list[bool]] = []

        if not self.conditional_score:
            for special_tokens_mask in batch["special_tokens_mask"]:
                batch["scoring_masks"].append([not special_token for special_token in special_tokens_mask])

        else:
            for input_ids, token_roles in zip(batch["input_ids"], batch["token_roles_internal"]):
                mask = [False] * len(input_ids)

                # Only set the mask to true where a token is part of the answer
                for i in token_roles["answer"]:
                    mask[i] = True

                batch["scoring_masks"].append(mask)

        return batch

    def create_masked_requests(
        self,
        batch: BatchEncoding,
    ) -> BatchEncoding:
        """Extend the existing batch and mask the relevant tokens based on the scoring mask."""

        mask_indices = self.mask_to_indices(batch["scoring_masks"])

        extended_batch = {
            k: torch.repeat_interleave(v, torch.tensor([i.size(0) for i in mask_indices], device=v.device), dim=0)
            for k, v in batch.items()
        }

        # Prepare tensors holding relevant information
        extended_batch["labels"] = torch.full(
            (extended_batch["input_ids"].size(0),),
            -1,
            dtype=torch.long,
        )
        extended_batch["masked_indices"] = torch.full(
            (extended_batch["input_ids"].size(0),),
            -1,
            dtype=torch.long,
        )
        extended_batch["statement_index"] = torch.full(
            (extended_batch["input_ids"].size(0),),
            -1,
            dtype=torch.long,
        )

        ### Apply the masking strategy ###
        statement_offset = 0  # to keep track which element in large batch to modify

        for statement_index, token_indices in enumerate(mask_indices):
            # Number of passes induced by this statement
            n = token_indices.size(0)

            # Passes within the extended_batch which belong to the current statement
            extended_batch_indices = torch.arange(statement_offset, statement_offset + n)

            extended_batch["labels"][extended_batch_indices] = extended_batch["input_ids"][
                extended_batch_indices, token_indices
            ]
            extended_batch["masked_indices"][extended_batch_indices] = token_indices

            # Store the statement index for each token such that it can later be assigned more easily
            extended_batch["statement_index"][extended_batch_indices] = statement_index

            # Mask tokens based on the selected masking-strategy
            if self.pll_metric == "original":
                # Mask each token individually
                extended_batch["input_ids"][extended_batch_indices, token_indices] = self.mask_token

            elif self.pll_metric == "within_word_l2r":
                word_ids = batch.word_ids(statement_index)

                # Mask each token and the remaineder of the same word
                for i, token_index in enumerate(token_indices):
                    current_word = word_ids[int(token_index.item())]

                    extended_batch["input_ids"][statement_offset + i, token_index] = self.mask_token

                    if current_word is None:
                        continue

                    # Go over all other tokens which are masked
                    for token_to_the_right in token_indices[i + 1 :]:
                        # and mask them if they are part of the same word
                        if word_ids[token_to_the_right] == current_word:
                            extended_batch["input_ids"][statement_offset + i, token_to_the_right] = self.mask_token
                        else:
                            # Assuming a word cannot be nested in another, different word means that all remaining
                            # tokens belong to other words as well
                            break

            elif self.pll_metric == "sentence_l2r":
                # For each pass, mask all tokens from the current token onward.
                for i, token_index in enumerate(token_indices):
                    start = int(token_index.item())
                    # Mask every token from the current token index to the end of the sequence.
                    extended_batch["input_ids"][statement_offset + i, start:] = self.mask_token

            elif self.pll_metric == "answer_l2r+word_l2r":
                if "token_roles_internal" not in batch:
                    msg = "'answer_l2r+word_l2r' requires the token roles to be set."
                    raise ValueError(msg)

                word_ids = batch.word_ids(statement_index)

                for i, token_index in enumerate(token_indices):
                    current_token_index = int(token_index.item())

                    answer_token_indices = batch["token_roles_internal"][statement_index]["answer"]
                    if current_token_index in answer_token_indices:
                        # Mask the remaining tokens of the answer
                        for selected_answer_index in answer_token_indices:
                            if selected_answer_index >= current_token_index:
                                extended_batch["input_ids"][statement_offset + i, selected_answer_index] = (
                                    self.mask_token
                                )
                    else:
                        # Mask each token and the remaineder of the same word
                        current_word = word_ids[int(token_index.item())]

                        extended_batch["input_ids"][statement_offset + i, token_index] = self.mask_token

                        if current_word is None:
                            continue

                        # Go over all other tokens which are masked
                        for token_to_the_right in token_indices[i + 1 :]:
                            # and mask them if they are part of the same word
                            if word_ids[token_to_the_right] == current_word:
                                extended_batch["input_ids"][statement_offset + i, token_to_the_right] = self.mask_token
                            else:
                                # Assuming a word cannot be nested in another, different word means that all remaining
                                # tokens belong to other words as well
                                break

            else:
                msg = f"PLL strategy {self.pll_metric} not implemented."
                raise NotImplementedError(msg)

            statement_offset += n

        return BatchEncoding(extended_batch)

    def process_extended_statements(
        self, large_batch: BatchEncoding, *, batch_size: Optional[int] = None
    ) -> Iterable[tuple[int, ScoredToken]]:
        """Process a stream of inputs in batches.

        Each input statement typically requires multiple inputs to the model.
        Since the exact number cannot be determined prior to tokenization, statements need to be tokenized already (and
        then extended) before being processed by the model.
        """
        batch_size = batch_size or self.batch_size

        for batch in iter_batches(large_batch, batch_size=batch_size):
            scored_indices = batch.pop("scored_indices")
            statement_indices = batch.pop("statement_index")
            batch_labels = batch.pop("labels")

            batch.pop("special_tokens_mask")
            batch.pop("length")

            # Add padding, make tensors
            model_input = self.tokenizer.pad(batch)
            model_input = model_input.to(self.device)

            # Forward the batches through the model
            with torch.no_grad():
                model_output = self.model(**model_input)

            # Shift so that tokens < n predict n
            batch_logits = model_output.logits
            batch_logits = batch_logits[torch.arange(batch_logits.size(0)), scored_indices].contiguous()

            batch_preds = torch.nn.functional.log_softmax(batch_logits, -1)
            batch_scores = batch_preds[torch.arange(batch_labels.size(0)), batch_labels].cpu()

            # Retrieve the score for each of the tokens
            for i, (statement_index, score, scored_index) in enumerate(
                zip(statement_indices, batch_scores, scored_indices)
            ):
                token = self.tokenizer.convert_ids_to_tokens(batch.tokens(i)[scored_index])
                yield statement_index, ScoredToken((token, score.item()))

    def mask_to_indices(self, scoring_masks: Sequence[ScoringMask]) -> list[torch.Tensor]:
        """Transform the scoring mask to a list of indices."""
        mask_indices = []
        # Replace the relevant tokens by the pad token
        for mask in scoring_masks:
            (indices,) = torch.where(torch.tensor(mask))
            mask_indices.append(indices)

        return mask_indices
