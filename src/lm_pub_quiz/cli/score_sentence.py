import logging
from dataclasses import dataclass
from typing import Union, cast

import cordage

from lm_pub_quiz.cli.config import ModelConfig
from lm_pub_quiz.evaluator import Evaluator

log = logging.getLogger(__name__)


@dataclass
class SentenceConfig:
    model: ModelConfig
    sentence: str
    device: Union[str, None] = None


def evaluate_sentence(config: SentenceConfig):
    log.info("Evaluating sentence '%s'", config.sentence)

    assert config.model.reduction != "tyq"

    evaluator: Evaluator = cast(Evaluator, config.model.create_evaluator(config.device))

    reduction = config.model.reduction if config.model.reduction != "none" else None
    # if existing, replace marker with itself
    result = evaluator.score_answers(
        template=config.sentence, answers=[evaluator.templater._answer_placeholder], reduction=reduction
    )[0]

    if reduction is not None:
        print(f"Resulting score: {result}")  # noqa: T201
    else:
        assert isinstance(result, tuple)
        token_scores, _ = result
        tokens, scores = zip(*token_scores)
        max_str_length = max(map(len, tokens))

        # Print header
        print(f"{'Rank':<{max_str_length+2}}{'Score':<8}")  # noqa: T201
        print("-" * (max_str_length + 10))  # noqa: T201
        for token, score in zip(tokens, scores):
            print(f"{token:<{max_str_length+2}}{score:>8.2f}")  # noqa: T201


def cli(args=None):
    """Run the command line interface."""
    logging.basicConfig(level=logging.WARNING)
    cordage.run(evaluate_sentence, args=args, config_only=True)
