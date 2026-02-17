from lm_pub_quiz.model_interface.base import ModelInterface, PLLModelInterfaceMixin
from lm_pub_quiz.model_interface.hf import HFModelInterface

MODEL_INTERFACE_CLASSES = {
    "hf": HFModelInterface,
}

__all__ = ["MODEL_INTERFACE_CLASSES", "ModelInterface", "PLLModelInterfaceMixin"]
