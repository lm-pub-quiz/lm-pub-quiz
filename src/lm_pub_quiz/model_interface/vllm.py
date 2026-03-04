import logging
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    BatchEncoding,
)

from lm_pub_quiz.model_interface.base import ModelInterface, PLLModelInterfaceMixin
from lm_pub_quiz.types import ScoringMask, StatementScore, TextRoles, TokenScoresAndRoles
from lm_pub_quiz.util import iter_batches

logger = logging.getLogger(__name__)


class VLLMInterface(PLLModelInterfaceMixin, ModelInterface):
    def __init__(
        self,
        model: str,
        *,
        loading_batch_size: int = 10000,
        ensure_bos_token_added: bool = True,
        use_tqdm: bool = False,
        **kw,
    ):
        self.model_name = model

        logger.debug("Creating vllm interface")

        from vllm import LLM  # noqa: PLC0415

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.llm = LLM(model, **kw)
        self.loading_batch_size = loading_batch_size
        self.ensure_bos_token_added = ensure_bos_token_added
        self.use_tqdm = use_tqdm

        self._bos_token_warning_issued: bool = False

    @classmethod
    def from_model(cls, model: Any, **kw) -> "VLLMInterface":
        """Create a `ModelInterface` from the given parameters.

        Load the model, tokenizer, etc. and instantiate a `ModelInterface`."""
        assert isinstance(model, str)

        return cls(model=model, **kw)

    @contextmanager
    def add_bos_token(self):
        _add_bos_token = self.tokenizer.add_bos_token
        self.tokenizer.add_bos_token = True
        yield
        self.tokenizer.add_bos_token = _add_bos_token

    def encode(
        self,
        statements: Sequence[str],
    ) -> tuple[BatchEncoding, Sequence[ScoringMask]]:
        """Encode the statements using the tokenizer and create an appropriate scoring mask.

        In case the conditional scores need to be created, set the scoring mask accordingly.
        """
        with self.add_bos_token():
            batch: BatchEncoding = self.tokenizer(
                statements,
                add_special_tokens=True,
                return_special_tokens_mask=True,
                return_attention_mask=True,
            )
            if batch["input_ids"][0][0] != self.tokenizer.bos_token_id:
                if self.ensure_bos_token_added:
                    # Issue a warning
                    if not self._bos_token_warning_issued:
                        self._bos_token_warning_issued = True
                        logger.warning(
                            "The tokenizer did not add a BOS-token, even with `add_bos_token=True`. "
                            "A BOS token was added manually, to prevent this (and keep the default behavior of the "
                            "tokenizer), set `ensure_bos_token_added=False`."
                        )

                    # Add the bos token manually
                    statements = [self.tokenizer.bos_token + s for s in statements]
                    batch = self.tokenizer(
                        list(statements),
                        add_special_tokens=True,
                        return_special_tokens_mask=True,
                        return_attention_mask=True,
                    )
                    for mask in batch["special_tokens_mask"]:
                        mask[0] = 1
                elif not self._bos_token_warning_issued:
                    self._bos_token_warning_issued = True
                    logger.info(
                        "The tokenizer did not add a BOS-token, even with `add_bos_token=True`. "
                        "To add the BOS token manually in this case, set `ensure_bos_token_added=True`."
                    )

        scoring_masks = [[not m for m in mask] for mask in batch["special_tokens_mask"]]

        return batch, scoring_masks

    def score_statements(
        self,
        statements: Iterable[str],
        *,
        text_roles: Iterable[TextRoles] | None = None,  # noqa: ARG002
        **kw,  # noqa: ARG002
    ) -> Iterable[StatementScore] | Iterable[TokenScoresAndRoles]:
        """Score individual texts (independent of the other options) using the Casual/Masked Language Model.

        Parameters:
            statements: The statements to score.
            text_roles: Which parts of the statement are the answer, template, and subject.

        Returns:
            Scores (or scores and roles) per statement
        """

        from vllm import SamplingParams, TokensPrompt  # noqa: PLC0415

        sampling_params = SamplingParams(
            temperature=0,
            prompt_logprobs=1,
            max_tokens=1,
            detokenize=False,
        )

        for prompts in iter_batches(statements, batch_size=self.loading_batch_size):
            encoded_prompts, _ = self.encode(list(prompts))

            request_outputs = self.llm.generate(
                [TokensPrompt(prompt_token_ids=token_ids) for token_ids in encoded_prompts["input_ids"]],
                sampling_params,
                # use_tqdm=self.use_tqdm,
                use_tqdm=partial(tqdm, leave=None),
            )

            for out in request_outputs:
                score = 0.0
                for token, logprobs in zip(out.prompt_token_ids, out.prompt_logprobs, strict=True):
                    if logprobs is not None:
                        score += logprobs[token].logprob

                yield score
