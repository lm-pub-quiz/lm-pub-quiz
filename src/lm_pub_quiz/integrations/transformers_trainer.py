import numpy as np
from transformers import TrainerCallback


class PubQuizCallback(TrainerCallback):
    """This callback handels on the fly BEAR evaluation for the huggingface Trainer class"""
    def __init__(self, trainer, evaluator, dataset, metrics=None, save_path=None, batch_size=32, template=0):
        """

        Args:
            trainer: required to call trainer logging functionality
            evaluator:
            dataset:
            metrics: one of [None, "overall", "domains", "cardinalities"]
            save_path: if specified, complete BEAR results for each evaluation are saved under the path
            batch_size:
            template: specify list of or single BEAR template indices to be used for evaluation
        """
        self.trainer = trainer
        self.evaluator = evaluator
        self.dataset = dataset
        self.save_path = save_path
        self.batch_size = batch_size
        self.template = template
        self.metrics = metrics

    def on_evaluate(self, args, state, control, **kwargs):

        result = self.evaluator.evaluate_dataset(self.dataset, template_index=self.template, batch_size=self.batch_size)
        if self.save_path:
            result.save(f"{self.save_path}/{state.epoch}_bear_results")

        overall_score = result.get_metrics(["accuracy", "support"], accumulate=True)
        metrics_data = {
            "eval_bear_score": overall_score
        }
        if self.metrics == "domains":
            domain_scores = result.get_metrics(["accuracy", "support"], accumulate="domains")
        elif self.metrics == "cardinality":
            cardinality_scores = result.get_metrics(["accuracy", "support"], accumulate="cardinality")

        self.trainer.log(metrics_data)
