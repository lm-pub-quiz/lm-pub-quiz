import logging
from dataclasses import dataclass
from typing import Union

import cordage

from lm_pub_quiz.cli.config import ModelConfig

log = logging.getLogger(__name__)


@dataclass
class AnswersConfig:
    model: ModelConfig
    template: str
    answers: str
    device: Union[str, None] = None


def rank_answers(config: AnswersConfig):
    """Rank answers for a template."""
    log.info("Evaluating template '%s'", config.template)

    evaluator = config.model.create_evaluator(config.device)

    answer_list = config.answers.split(",")

    reduction = config.model.reduction if config.model.reduction != "none" else None
    evaluator.evaluate_instance(template=config.template, answers=answer_list, print_ranking=True, reduction=reduction)


def cli(args=None):
    """Run the command line interface."""
    logging.basicConfig(level=logging.WARNING)
    cordage.run(rank_answers, args=args, config_only=True)
