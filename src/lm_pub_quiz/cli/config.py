from dataclasses import dataclass
from typing import Optional, Type, Union

from lm_pub_quiz import Evaluator
from lm_pub_quiz.evaluator import BaseEvaluator
from lm_pub_quiz.tyq_evaluator import TyQEvaluator


@dataclass
class ModelConfig:
    """Configuration used to initialize an evaluator object."""

    name_or_path: str
    lm_type: Optional[str] = None
    tokenizer: Optional[str] = None
    reduction: str = "sum"
    pll_metric: Optional[str] = None

    def create_evaluator(self, device: Union[str, int, None] = None) -> BaseEvaluator:
        kw = {}

        if self.pll_metric is not None:
            kw["pll_metric"] = self.pll_metric

        if self.lm_type is not None:
            kw["model_type"] = self.lm_type

        cls: Union[Type[Evaluator], Type[TyQEvaluator]]
        if self.reduction != "tyq":
            cls = Evaluator
        else:
            cls = TyQEvaluator

        return cls.from_model(model=self.name_or_path, tokenizer=self.tokenizer, device=device, **kw)
