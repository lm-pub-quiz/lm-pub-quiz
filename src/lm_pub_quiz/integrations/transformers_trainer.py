from transformers import TrainerCallback


class PubQuizCallback(TrainerCallback):
    """This callback handels on the fly LM Pub Quiz evaluation for the huggingface Trainer class"""

    def __init__(
        self, trainer, evaluator, dataset, accumulate=None, save_path=None, batch_size=32, template_index=None
    ):
        """

        Args:
            trainer: required to call trainer logging functionality
            evaluator:
            dataset:
            accumulate: one of [None, "domains", "cardinality"]
            save_path: if specified, complete LM Pub Quiz results for each evaluation are saved under the path
            batch_size:
            template: specify list of or single LM Pub Quiz template indices to be used for evaluation
        """
        self.trainer = trainer
        self.evaluator = evaluator
        self.dataset = dataset
        self.save_path = save_path
        self.batch_size = batch_size
        self.template_index = template_index
        self.accumulate = accumulate
        self.report_name = dataset.name.replace(" ", "_") if dataset.name is not None else "lm_pub_quiz"

        # check dataset for compatibility with requested metrics
        if isinstance(self.accumulate, str):
            try:
                dataset[0].relation_info(self.accumulate)
            except KeyError as e:
                error_msg = f"Cannot retrieve {self.accumulate} results without relevant info in dataset."
                raise ValueError(error_msg) from e

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: ARG002
        metrics_data = {}

        result = self.evaluator.evaluate_dataset(
            self.dataset, template_index=self.template_index, batch_size=self.batch_size
        )
        if self.save_path:
            result.save(f"{self.save_path}/{state.epoch}_{self.report_name}_results")

        overall_score = result.get_metrics(["accuracy", "support"], accumulate=True)
        metrics_data[f"eval_{self.report_name}_score"] = overall_score["accuracy"]

        if isinstance(self.accumulate, str):
            accumulated_scores = result.get_metrics(["accuracy", "support"], accumulate=self.accumulate)
            for metric, score in accumulated_scores["accuracy"].items():
                metrics_data[f"eval_{self.report_name}_{metric.replace(' ', '_')}"] = score

        self.trainer.log(metrics_data)
