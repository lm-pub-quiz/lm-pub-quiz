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
            metrics: one of [None, "overall", "domains", "cardinality"]
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

        # check dataset for compatibility with requested metrics
        if self.metrics == "domains":
            try:
                dataset[0].relation_info("domains")
            except KeyError as e:
                raise ValueError(f"Cannot retrieve domain results without domain info in dataset.")

    def on_evaluate(self, args, state, control, **kwargs):
        if isinstance(self.template, int):
            templates = [self.template]
            flag_template = False
        else:
            templates = self.template
            flag_template = True

        metrics_data = dict()
        for template in templates:
            template_suffix = "_" + str(template) if flag_template else ""
            result = self.evaluator.evaluate_dataset(self.dataset, template_index=self.template, batch_size=self.batch_size)
            if self.save_path:
                result.save(f"{self.save_path}/{state.epoch}_bear_results{template_suffix}")

            overall_score = result.get_metrics(["accuracy", "support"], accumulate=True)
            metrics_data[f"eval_bear_score{template_suffix}"] = overall_score["accuracy"]

            if self.metrics == "domains":
                domain_scores = result.get_metrics(["accuracy", "support"], accumulate="domains")
                for domain, score in domain_scores["accuracy"].items():
                    metrics_data[f"eval_bear_{domain}{template_suffix}"] = score
            elif self.metrics == "cardinality":
                cardinality_scores = result.get_metrics(["accuracy", "support"], accumulate="cardinality")
                for cardinality, score in cardinality_scores["accuracy"].items():
                    metrics_data[f"eval_bear_{cardinality.replace(' ', '_')}{template_suffix}"] = score

        self.trainer.log(metrics_data)
