from lm_pub_quiz.model_interface.base import ModelInterface, PLLModelInterfaceMixin
from lm_pub_quiz.model_interface.hf import HFModelInterface
from lm_pub_quiz.model_interface.vllm import VLLMInterface

MODEL_INTERFACE_CLASSES = {"hf": HFModelInterface, "vllm": VLLMInterface}

__all__ = ["MODEL_INTERFACE_CLASSES", "ModelInterface", "PLLModelInterfaceMixin"]
