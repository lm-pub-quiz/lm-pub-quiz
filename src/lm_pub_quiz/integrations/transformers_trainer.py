import numpy as np
from transformers import TrainerCallback


class PubQuizCallback(TrainerCallback):
    """This callback handels on the fly BEAR evaluation for the huggingface Trainer class"""
    def __init__(self, evaluator, dataset, metrics=None, save_path=None, batch_size=32, template=0):
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

        bear_metrics = result.get_metrics(self.metrics)
        weighted_accuracy = np.average(bear_metrics.accuracy, weights=bear_metrics.num_instances)

        # TODO: Include additional metrics
        metrics_data = {
            "eval_bear_score": weighted_accuracy
        }

        self.trainer.log(metrics_data)
