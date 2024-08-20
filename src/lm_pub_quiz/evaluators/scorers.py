from abc import abstractmethod
from typing import Dict, List, Optional, Sequence

import torch
from transformers import BatchEncoding

from lm_pub_quiz.evaluators.model_util import ModelMixin
from lm_pub_quiz.evaluators.util import iter_batches
from lm_pub_quiz.types import ScoringMask


class PLLScorerBase(ModelMixin):
    """This class is used retrieve PLL scores for tokens or the complete statement."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)

    @abstractmethod
    def score_statements(
        self,
        batched_statements: BatchEncoding,
        scoring_masks: Optional[Sequence[ScoringMask]],
        batch_size: int = 1,
    ) -> List[List[float]]:
        """Compute the PLL score for the tokens (determined by the scoring mask) in a statements.

        This function must be implemented by child-classes for each model-type.
        """


class MaskedLMScorer(PLLScorerBase):
    def __init__(self, pll_metric: str = "within_word_l2r", **kw) -> None:
        if pll_metric not in ("original", "within_word_l2r"):
            msg = f"PLL strategy {pll_metric} not know."
            raise ValueError(msg)

        super().__init__(**kw)
        self.pll_metric = pll_metric

    @property
    def mask_token(self) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        assert isinstance(token_id, int)
        return token_id

    def score_statements(
        self,
        batch: BatchEncoding,
        scoring_masks: Optional[Sequence[ScoringMask]],
        batch_size: int = 1,
    ) -> List[List[float]]:

        if scoring_masks is None:
            scoring_masks = [[True] * batch["input_ids"].size(1) for _ in range(batch["input_ids"].size(0))]

        # Extend the existing batch into a larger batch based on the scoring mask
        # and mask the relevant tokens
        extended_batch = self.create_masked_batch(batch, scoring_masks)

        # Split up the larger batch based on the batch size
        # and forward the batches through the model

        token_scores: List[List[float]] = [[] for _ in range(batch["input_ids"].size(0))]

        for sub in iter_batches(extended_batch, batch_size=batch_size):
            batch_labels = sub.pop("labels")
            masked_indices = sub.pop("masked_indices")
            batch_index = sub.pop("batch_index")

            with torch.no_grad():
                del sub["special_tokens_mask"]
                del sub["length"]
                model_output = self.model(**sub)

            # Shift so that tokens < n predict n
            batch_logits = model_output.logits
            batch_logits = batch_logits[torch.arange(batch_logits.size(0)), masked_indices].contiguous()

            batch_preds = torch.nn.functional.log_softmax(batch_logits, -1)
            batch_scores = batch_preds[torch.arange(batch_labels.size(0)), batch_labels]

            # Retrieve the score for each of the tokens
            for i, score in zip(batch_index, batch_scores):
                token_scores[i].append(score.item())

        return token_scores

    def mask_to_indices(self, scoring_masks: Sequence[ScoringMask]) -> List[torch.Tensor]:
        mask_indices = []
        # Replace the relevant tokens by the pad token
        for mask in scoring_masks:
            (indices,) = torch.where(torch.tensor(mask))
            mask_indices.append(indices)

        return mask_indices

    def create_masked_batch(
        self, batch: BatchEncoding, scoring_masks: Sequence[ScoringMask]
    ) -> Dict[str, torch.Tensor]:
        mask_indices = self.mask_to_indices(scoring_masks)

        extended_batch = {
            k: torch.repeat_interleave(v, torch.tensor([i.size(0) for i in mask_indices]), dim=0)
            for k, v in batch.items()
        }

        extended_batch["labels"] = torch.full((extended_batch["input_ids"].size(0),), -1, dtype=torch.long)
        extended_batch["masked_indices"] = torch.full((extended_batch["input_ids"].size(0),), -1, dtype=torch.long)
        extended_batch["batch_index"] = torch.full((extended_batch["input_ids"].size(0),), -1, dtype=torch.long)

        # Apply the masking strategy
        offset = 0  # to keep track which element in large batch to modify

        for original_batch_index, token_indices in enumerate(mask_indices):
            n = token_indices.size(0)
            extended_batch_indices = torch.arange(offset, offset + n)

            extended_batch["labels"][extended_batch_indices] = extended_batch["input_ids"][
                extended_batch_indices, token_indices
            ]
            extended_batch["masked_indices"][extended_batch_indices] = token_indices
            extended_batch["masked_indices"][extended_batch_indices] = original_batch_index

            if self.pll_metric == "original":
                extended_batch["input_ids"][extended_batch_indices, token_indices] = self.mask_token

            elif self.pll_metric == "within_word_l2r":
                word_ids = batch.word_ids(original_batch_index)

                for extended_batch_index, token_index in zip(extended_batch_indices, token_indices):
                    # Go over all following tokens and mask them if they belong to the same word
                    current_word = word_ids[int(token_index.item())]

                    within_word_right_tokens = [
                        i
                        for i in range(int(token_index.item()), extended_batch["input_ids"].size(1))
                        if (word_ids[i] is not None) and (word_ids[i] == current_word)
                    ]

                    extended_batch["input_ids"][extended_batch_index, within_word_right_tokens] = self.mask_token

            else:
                msg = f"PLL strategy {self.pll_metric} not implemented."
                raise NotImplementedError(msg)

            offset += n

        return extended_batch


class CausalLMScorer(PLLScorerBase):
    def score_statements(
        self,
        batch: BatchEncoding,
        scoring_masks: Optional[Sequence[Optional[ScoringMask]]],
        batch_size: int = 1,
    ) -> List[List[float]]:

        scores: List[List[float]] = []

        if scoring_masks is None:
            scoring_masks = [None] * batch["input_ids"].size(0)

        for subbatch, masks in zip(
            iter_batches(batch, batch_size=batch_size),  # Iter through the batches
            iter_batches(scoring_masks, batch_size=batch_size),  # and masks at the same time
        ):
            # Forward the batch through the model
            with torch.no_grad():
                model_output = self.model(subbatch)

            # Shift so that tokens < n predict n
            batch_logits = model_output.logits[:, :-1]
            batch_labels = subbatch["input_ids"][:, 1:]

            for i, mask in enumerate(masks):
                if mask is None:
                    logits = batch_logits[i, :].contiguous()
                    labels = batch_labels[i, :].contiguous()
                else:
                    logits = batch_logits[i, mask].contiguous()
                    labels = batch_labels[i, mask].contiguous()

                preds = torch.nn.functional.log_softmax(logits, -1)

                scores.append(preds[labels.size(0), labels].tolist())

        return scores
