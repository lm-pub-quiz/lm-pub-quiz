import logging
from typing import Literal, Optional, Tuple, Type, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from typing_extensions import Self

from lm_pub_quiz.evaluators.util import BaseMixin

log = logging.getLogger(__name__)


class ModelMixin(BaseMixin):
    _mlm_keywords: Tuple[str, ...] = ("bert",)
    _clm_keywords: Tuple[str, ...] = ("opt", "gpt", "llama", "bloom", "google/gemma", "mistral")

    def __init__(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        device: Union[torch.device, int, str, None],
        **kw,
    ):
        super().__init__(**kw)
        self.model_name: str = self._get_model_name(model)

        self.model = model
        self.tokenizer = tokenizer
        self.device = self._get_device(device)

    @classmethod
    def _get_device(cls, device_input: Union[torch.device, int, str, None]) -> torch.device:
        if device_input is None:
            try:
                return torch.get_default_device()
            except AttributeError:
                return torch.device("cpu")
        elif isinstance(device_input, torch.device):
            return device_input
        elif isinstance(device_input, int):
            return torch.device(f"cuda:{device_input}")
        else:
            return torch.device(device_input)

    @classmethod
    def _get_model_name(cls, model: Union[str, PreTrainedModel]) -> str:
        if isinstance(model, str):
            return model
        else:
            return model.base_model_prefix

    @classmethod
    def _get_model(
        cls, model: Union[str, PreTrainedModel], model_type: Optional[str] = None, device: Optional[str] = None, **kw
    ) -> PreTrainedModel:
        if not isinstance(model, str):
            return model

        if model_type is None:
            model_type = cls._infer_type_from_name(model)
            log.debug("Inferred type of model `%s`: %s", model, model_type)

        model_class: Type

        if model_type == "MLM":
            model_class = AutoModelForMaskedLM
        elif model_type == "CLM":
            model_class = AutoModelForCausalLM
        else:
            msg = f"Unkown model type: '{model_type}'."
            log.error(msg)
            raise ValueError(msg)

        return model_class.from_pretrained(model, device_map=device, return_dict=True, **kw)

    @staticmethod
    def _get_tokenizer(
        model: Union[str, PreTrainedModel], tokenizer: Union[str, PreTrainedTokenizerFast, None]
    ) -> PreTrainedTokenizerFast:
        """Retrieve a tokenizer that matches the model or tokenizer string."""
        if tokenizer is None:
            if isinstance(model, str):
                return AutoTokenizer.from_pretrained(model, use_fast=True)
            else:
                return AutoTokenizer.from_pretrained(model.config.name_or_path, use_fast=True)
        elif isinstance(tokenizer, str):
            return AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        else:
            return tokenizer

    @classmethod
    def _infer_type_from_name(cls, model: str) -> Literal["MLM", "CLM"]:
        """Infer the type of model (MLM or CLM) based on the model name."""

        if any(k in model.lower() for k in cls._mlm_keywords):
            return "MLM"
        elif any(k in model.lower() for k in cls._clm_keywords):
            return "CLM"
        else:
            msg = f"Cannot infer model type from the `model_name_or_path`: '{model}'."
            log.error(msg)
            raise ValueError(msg)

    @classmethod
    def _infer_type_from_object(cls, model: PreTrainedModel):
        """Infer the type of model (MLM or CLM) based on model object."""
        if model.config.is_decoder:
            return "CLM"
        else:
            return "MLM"

    @classmethod
    def from_model(
        cls,
        model: Union[str, PreTrainedModel],
        *,
        model_type: Optional[str] = None,
        device: Union[torch.device, None, str, int] = None,
        **kw,
    ) -> Self:

        model = cls._get_model(model=model, model_type=model_type, **kw)
        tokenizer = cls._get_tokenizer(model=model, tokenizer=kw.pop("tokenizer", None))

        return cls(model=model, tokenizer=tokenizer, model_type=model_type, device=cls._get_device(device), **kw)