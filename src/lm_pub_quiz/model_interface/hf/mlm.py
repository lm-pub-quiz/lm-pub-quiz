import copy
import itertools
import logging
from collections.abc import Iterable
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
from lm_pub_quiz.types import ScoredToken, StatementScore, T, TextRoles, TokenScoresAndRoles
from lm_pub_quiz.util import iter_batches

tqdm.pandas()


log = logging.getLogger(__name__)


class MLMInterface(PLLModelInterfaceMixin, HFModelInterface):
    def __init__(
        self,
        *args,
        pll_metric: str = "within_word_l2r",
        conditional_score: bool = False,
        preprocessing_batch_size: int = 1000,
        **kw,
    ) -> None:
        if pll_metric not in ("original", "within_word_l2r", "sentence_l2r", "answer_l2r+word_l2r"):
            msg = f"PLL strategy {pll_metric} not know."
            raise KeyError(msg)

        super().__init__(*args, **kw)

        self.pll_metric = pll_metric
        self.conditional_score = conditional_score
        self.preprocessing_batch_size = preprocessing_batch_size

    def get_metadata(self) -> dict[str, Any]:
        return {
            "pll_metric": self.pll_metric,
            "conditional_score": self.conditional_score,
            "preprocessing_batch_size": self.preprocessing_batch_size,
            **super().get_metadata(),
        }

    @property
    def mask_token(self) -> int:
        """Return the mask token id used by the tokenizer."""

        token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        assert isinstance(token_id, int)
        return token_id

    def score_statements(
        self,
        statements: Iterable[str],
        *,
        text_roles: Optional[Iterable[TextRoles]] = None,
        batch_size: Optional[int] = None,
        **kw,
    ) -> Union[Iterable[TokenScoresAndRoles], Iterable[StatementScore]]:
        """Score individual texts (independent of the other options) using the Masked Language Model.

        Parameters:
            statements: The statements to score.
            text_roles: Which parts of the statement are the answer, template, and subject.

        Returns:
            Scores (or scores and roles) per statement
        """
        if text_roles is None and self.conditional_score:
            msg = "`roles` need to be set to use `conditional_score`."
            raise ValueError(msg)

        # Tokenize, translate the text roles to token roles, and create scoring mask
        if text_roles is None:
            preprocessed_statements = (
                self.preprocess_statements(list(batch_statements))
                for batch_statements in iter_batches(statements, batch_size=self.preprocessing_batch_size)
            )
        else:
            preprocessed_statements = (
                self.preprocess_statements(list(batch_statements), text_roles=list(batch_text_roles))
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

        def get_yield_value(large_batch, i, scored_tokens):
            ### Yield the return values ###
            if reduction is None:
                if "token_roles_internal" in large_batch:
                    token_roles_output = remap_token_roles(
                        token_roles_internal=large_batch["token_roles_internal"][i],
                        scoring_mask=large_batch["scoring_masks"][i],
                    )
                else:
                    token_roles_output = {}

                return scored_tokens, token_roles_output

            else:
                return reduction_func([score for token, score in scored_tokens])

        # Process the extended requests
        for large_batch in masked_requests:
            statement_index: int = 0
            scored_tokens: list[ScoredToken] = []

            new_statement_index: int
            new_scored_token: ScoredToken

            for i, (new_statement_index, new_scored_token) in enumerate(
                self.process_extended_statements(large_batch, batch_size=batch_size)
            ):
                if statement_index != new_statement_index:
                    assert new_statement_index == statement_index + 1

                    value = get_yield_value(large_batch, i - 1, scored_tokens)

                    log.debug(
                        "Statement %d: %s => %s",
                        statement_index,
                        " ".join(f"{t} ({s:.2f})" for t, s in scored_tokens),
                        str(value),
                    )

                    yield value

                    scored_tokens = []
                    statement_index = new_statement_index

                scored_tokens.append(new_scored_token)

            if len(scored_tokens) > 0:
                value = get_yield_value(large_batch, -1, scored_tokens)

                log.debug(
                    "Statement %d: %s => %s",
                    statement_index,
                    " ".join(f"{t} ({s:.2f})" for t, s in scored_tokens),
                    str(value),
                )

                yield value

    def preprocess_statements(
        self, statements: list[str], *, text_roles: Optional[list[TextRoles]] = None
    ) -> BatchEncoding:
        """Tokenize statements, translate text roles (char level) to token roles and determine which tokens to score."""
        if text_roles is None and self.conditional_score:
            msg = "`roles` need to be set to use `conditional_score`."
            raise ValueError(msg)

        batch = self.tokenizer(
            list(statements),
            padding=False,
            return_special_tokens_mask=True,
        )

        if text_roles is not None:
            token_roles = derive_token_roles_internal(batch=batch, text_roles=text_roles)
            batch["token_roles_internal"] = token_roles

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

        # Get the indices of tokens to be scored
        mask_indices = [[i for i, m in enumerate(mask) if m] for mask in batch["scoring_masks"]]

        def repeated_foreach_scored_token(data: Iterable[list[T]], ns: Iterable[int]) -> list[list[T]]:
            return [
                copy.deepcopy(element)
                for element in itertools.chain.from_iterable(
                    itertools.repeat(base_element, n) for base_element, n in zip(data, ns)
                )
            ]

        ns = [len(indices) for indices in mask_indices]

        extended_batch = BatchEncoding({k: repeated_foreach_scored_token(v, ns) for k, v in batch.items()})

        # Prepare field holding relevant information
        extended_batch["labels"]: list[int] = []
        extended_batch["scored_tokens"]: list[int] = []
        extended_batch["statement_index"]: list[int] = []

        ### Apply the masking strategy ###
        extended_batch_index: int  # to keep track which element in large batch to modify
        extended_batch_index_counter = itertools.count()

        for statement_index, token_indices in enumerate(mask_indices):
            scoring_index: int  # index of the current scored token (within the list of scored tokens)
            token_index: int  # index of the token within the current statement

            for scoring_index, (token_index, extended_batch_index) in enumerate(
                zip(token_indices, extended_batch_index_counter)
            ):
                # Store the statement index for each token such that it can later be assigned more easily
                extended_batch["statement_index"].append(statement_index)

                # Current token index being scored
                extended_batch["scored_tokens"].append(token_index)

                # Label for the current token index
                extended_batch["labels"].append(batch["input_ids"][statement_index][token_index])

                input_ids = extended_batch["input_ids"][extended_batch_index]

                # Mask tokens based on the selected masking-strategy
                if self.pll_metric == "original":
                    # Mask each token individually
                    input_ids[token_index] = self.mask_token

                elif self.pll_metric == "within_word_l2r":
                    # Mask each token and the remaineder of the same word
                    input_ids[token_index] = self.mask_token

                    word_ids = batch.word_ids(statement_index)
                    current_word = word_ids[token_index]

                    if current_word is None:
                        continue

                    # Go over all other tokens which are masked
                    for token_to_the_right in token_indices[scoring_index + 1 :]:
                        # and mask them if they are part of the same word
                        if word_ids[token_to_the_right] == current_word:
                            input_ids[token_to_the_right] = self.mask_token
                        else:
                            # Assuming a word cannot be nested in another, different word means that all remaining
                            # tokens belong to other words as well
                            break

                elif self.pll_metric == "sentence_l2r":
                    # Mask every token (including non-scored) from the current token index to the end of the sequence.
                    for i in range(token_indices[scoring_index], len(input_ids)):
                        input_ids[i] = self.mask_token

                elif self.pll_metric == "sentence_l2r-scored_only":
                    # Mask every scored token from the current token index to the end of the sequence.
                    for i in token_indices[scoring_index:]:
                        input_ids[i] = self.mask_token

                elif self.pll_metric == "answer_l2r+word_l2r":
                    if "token_roles_internal" not in batch:
                        msg = "'answer_l2r+word_l2r' requires the token roles to be set."
                        raise ValueError(msg)

                    word_ids = batch.word_ids(statement_index)

                    answer_token_indices = batch["token_roles_internal"][statement_index]["answer"]

                    if token_index in answer_token_indices:
                        # Mask the remaining tokens of the answer
                        for i in answer_token_indices:
                            if i >= token_index:
                                input_ids[i] = self.mask_token

                    else:
                        # Mask each token and the remaineder of the same word
                        input_ids[token_index] = self.mask_token

                        word_ids = batch.word_ids(statement_index)
                        current_word = word_ids[token_index]

                        if current_word is None:
                            continue

                        # Go over all other tokens which are masked
                        for token_to_the_right in token_indices[scoring_index + 1 :]:
                            # and mask them if they are part of the same word
                            if word_ids[token_to_the_right] == current_word:
                                input_ids[token_to_the_right] = self.mask_token
                            else:
                                # Assuming a word cannot be nested in another, different word means that all remaining
                                # tokens belong to other words as well
                                break
                else:
                    msg = f"PLL strategy {self.pll_metric} not implemented."
                    raise NotImplementedError(msg)

        return extended_batch

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
            scored_tokens = batch.pop("scored_tokens")
            statement_indices = batch.pop("statement_index")
            batch_labels = torch.tensor(batch.pop("labels"))

            batch.pop("special_tokens_mask")
            batch.pop("token_roles_internal", None)
            batch.pop("scoring_masks")
            # batch.pop("length")

            # Add padding, make tensors
            model_input = self.tokenizer.pad(batch, return_tensors="pt")
            model_input = model_input.to(self.device)

            # Forward the batches through the model
            with torch.no_grad():
                model_output = self.model(**model_input)

            batch_logits = model_output.logits
            batch_logits = batch_logits[torch.arange(batch_logits.size(0)), scored_tokens].contiguous()

            batch_preds = torch.nn.functional.log_softmax(batch_logits, -1)
            batch_scores = batch_preds[torch.arange(batch_labels.size(0)), batch_labels].cpu()

            # Retrieve the score for each of the tokens
            for statement_index, score, token in zip(
                statement_indices, batch_scores, self.tokenizer.convert_ids_to_tokens(batch_labels)
            ):
                yield statement_index, ScoredToken((token, score.item()))
