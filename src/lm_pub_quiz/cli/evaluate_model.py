import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cordage

from lm_pub_quiz import Dataset
from lm_pub_quiz.cli.config import ModelConfig
from lm_pub_quiz.evaluator import BaseEvaluator, MaskedLMEvaluator

log = logging.getLogger(__name__)


@dataclass
class Configuration:
    model: ModelConfig
    dataset_path: Path
    output_base_path: Path = Path("outputs")
    debug: bool = field(default=False, metadata={"help": "Only use 2 instances per relation (if true)."})
    device: Union[str, None] = None
    batch_size: int = 32
    relation: Optional[str] = None
    template_index: int = 0


def evaluate_model(config: Configuration):
    """Evaluate a given model on a dataset."""
    # Load dataset
    dataset = Dataset.from_path(config.dataset_path)

    # Create Evaluator (and load model)
    evaluator: BaseEvaluator = config.model.create_evaluator(config.device)

    reduction = config.model.reduction if config.model.reduction != "none" else None

    # Save the result
    save_path = config.output_base_path

    if dataset.name is not None:
        save_path = save_path / dataset.name

    retrieval_part: str
    if isinstance(evaluator, MaskedLMEvaluator):
        retrieval_part = f"{evaluator.pll_metric}_{config.model.reduction}"
    else:
        retrieval_part = f"{config.model.reduction}"

    save_path = save_path / Path(config.model.name_or_path).name / retrieval_part

    log.info("Saving result at '%s'.", str(save_path))

    # Run evaluation
    subsample = 2 if config.debug else None

    if config.relation is None:
        evaluator.evaluate_dataset(
            dataset,
            subsample=subsample,
            batch_size=config.batch_size,
            save_path=save_path,
            reduction=reduction,
            template_index=config.template_index,
        )
    else:
        relation = dataset[config.relation]
        log.info("Starting evaluation of relation %s.", relation)
        relation_result = evaluator.evaluate_relation(
            relation,
            template_index=config.template_index,
            batch_size=config.batch_size,
            subsample=subsample,
        )
        evaluator.update_result_metadata(relation_result, dataset=dataset)
        rel_path = relation_result.save(save_path)
        log.info("Saved relation results at: %s.", str(rel_path))


def cli(args=None):
    """Run the command line interface."""
    logging.basicConfig(level=logging.INFO)
    cordage.run(evaluate_model, args=args, config_only=True)
