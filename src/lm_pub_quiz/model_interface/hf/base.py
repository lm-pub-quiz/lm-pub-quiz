import inspect
import logging
from typing import Any, Literal, Optional, Union

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from typing_extensions import Self

from lm_pub_quiz.model_interface import hf
from lm_pub_quiz.model_interface.base import ModelInterface

log = logging.getLogger(__name__)


class HFModelInterface(ModelInterface):
    _mlm_keywords: tuple[str, ...] = ("bert",)
    _clm_keywords: tuple[str, ...] = ("opt", "gpt", "llama", "bloom", "google/gemma", "mistral")

    model_name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    device: torch.device

    def __init__(
        self,
        *,
        model: PreTrainedModel,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        model_kw: Optional[dict[str, Any]] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        device: Union[torch.device, int, str, None] = None,
        batch_size: int = 1,
        **kw,
    ):
        super().__init__(**kw)

        model_kw = model_kw or {}
        if "device" not in model_kw:
            model_kw["device"] = device

        self.model_name = model_name or self._get_model_name(model)
        self.model = self._get_model(model=model, model_type=model_type, **model_kw)
        self.tokenizer = self._get_tokenizer(model=model, tokenizer=tokenizer)
        self.device = self._get_device(device)
        self.batch_size = batch_size

    @classmethod
    def from_model(
        cls,
        model: Union[str, PreTrainedModel],
        model_type: Optional[str] = None,
        **kw,
    ) -> Self:
        """Create an interface for the given model.

        In some cases, the model type can be derived from the model itself. To ensure
        the right type is chosen, it's recommended to set `model_type` manually.

        Parameters:
            model str | PreTrainedModel: The model to evaluate.
            model_type str | None: The type of model (determines the scoring scheme to be used).

        Returns:
            HFPLLModelInterface: The evaluator instance suitable for the model.
        """

        if not inspect.isabstract(cls):
            return cls(model=model, model_type=model_type, **kw)

        if model_type is None:
            model_type = cls._infer_model_type(model)

        interface_class: type[HFModelInterface]

        if model_type == "MLM":
            interface_class = hf.MLMInterface
        elif model_type == "CLM":
            interface_class = hf.CLMInterface
        else:
            msg = f"The model type {model_type} is not implemented."
            raise ValueError(msg)

        return interface_class.from_model(model=model, model_type=model_type, **kw)  # type: ignore (currently not handled correctly?)

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
        cls,
        model: Union[str, PreTrainedModel],
        model_type: Optional[str] = None,
        device: Optional[torch.device] = None,
        **kw,
    ) -> PreTrainedModel:
        if not isinstance(model, str):
            return model

        if model_type is None:
            model_type = cls._infer_model_type(model)
            log.debug("Inferred type of model `%s`: %s", model, model_type)

        model_class: type[AutoModel]

        if model_type == "MLM":
            model_class = AutoModelForMaskedLM
        elif model_type == "CLM":
            model_class = AutoModelForCausalLM
        else:
            msg = f"Unkown model type: '{model_type}'."
            log.error(msg)
            raise ValueError(msg)

        if "device_map" not in kw and device is not None:
            kw["device_map"] = device

        model = model_class.from_pretrained(model, return_dict=True, **kw)

        if "device_map" not in kw and device is not None:
            model.to(device)

        return model

    @staticmethod
    def _get_tokenizer(
        model: Union[str, PreTrainedModel], tokenizer: Union[str, PreTrainedTokenizerFast, None]
    ) -> PreTrainedTokenizerFast:
        """Retrieve a tokenizer that matches the model or tokenizer string."""
        if tokenizer is None:
            if isinstance(model, str):
                tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path, use_fast=True)
        elif isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                log.info("Tokenizer has no PAD token: Using the EOS token instead.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                msg = "Tokenizer has not PAD or EOS token."
                raise ValueError(msg)

        return tokenizer

    @classmethod
    def _infer_model_type(cls, model: str) -> Literal["MLM", "CLM"]:
        """Infer the type of model (MLM or CLM) based on the model name."""

        if isinstance(model, str):
            if any(k in model.lower() for k in cls._mlm_keywords):
                return "MLM"
            elif any(k in model.lower() for k in cls._clm_keywords):
                return "CLM"
            else:
                msg = f"Cannot infer model type from the `model_name_or_path`: '{model}'."
                log.error(msg)
                raise ValueError(msg)
        else:
            return cls._infer_model_type(model.config.model_type)
