import numpy as np
from transformers import TrainerCallback

from lm_pub_quiz.data import Dataset
from lm_pub_quiz.evaluator import Evaluator


class PubQuizCallback(TrainerCallback):
    """This callback handels on the fly BEAR evaluation for the huggingface Trainer class"""
    def __init__(self, trainer, dataset_path, model_type=None, save_path=None, batch_size=32, template=0):
        self.trainer = trainer
        self.model_type = model_type
        self.save_path = save_path
        self.batch_size = batch_size
        self.template = template
        self.dataset = Dataset.from_path(dataset_path)

    def on_evaluate(self, args, state, control, **kwargs):
        evaluator = Evaluator.from_model(
            kwargs['model'],
            model_type=self.model_type,
            tokenizer=kwargs['tokenizer'],
            device=kwargs['model'].device,
        )
        result = evaluator.evaluate_dataset(self.dataset, template_index=self.template, batch_size=self.batch_size)
        if self.save_path:
            result.save(f"{self.save_path}/{state.epoch}_bear_results")

        bear_metrics = result.get_metrics(["accuracy", "num_instances"])
        weighted_accuracy = np.average(bear_metrics.accuracy, weights=bear_metrics.num_instances)

        metrics = {
            'eval_bear_score': weighted_accuracy
        }

        self.trainer.log(metrics)
