import logging

import pytest
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from lm_pub_quiz import Dataset, Evaluator
from lm_pub_quiz.integrations import PubQuizCallback

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.log_data = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
        if logs is not None:
            self.log_data.append(logs)


def test_callback(request, tmp_path):
    probing_dataset = Dataset.from_path(
        request.path.parent / "test_data" / "dummy_dataset",
        relation_info=request.path.parent / "test_data" / "dummy_relation_info.json",
    )

    # Since the model will be modified, we will not use the model cache
    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased", clean_up_tokenization_spaces=True)

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]], truncation=True, padding="max_length", max_length=512)

    mlm_eval_dataset = HFDataset.from_dict(
        {
            "text": ["I will not waste chalk.", "I will not skateboard in the halls."]
            + ["I will not cut corners."] * 2,
        }
    ).map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    mlm_train_dataset = HFDataset.from_dict({"text": ["I will not bury the new kid."] * 2}).map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        tmp_path,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        num_train_epochs=5.0,
        logging_first_step=True,
        logging_steps=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to="none",
        learning_rate=1e-4,
    )

    logging_callback = LoggingCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mlm_train_dataset,
        eval_dataset=mlm_eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[logging_callback],
    )

    evaluator = Evaluator.from_model(
        model,
        tokenizer=tokenizer,
        model_type="MLM",
        batch_size=1,
        dataset=probing_dataset,
        template_index=0,
    )

    assert evaluator.evaluate_dataset(probing_dataset).get_metrics("accuracy", accumulate=True)["accuracy"] == 1.0

    probing_dataset = Dataset.from_path(
        request.path.parent / "test_data" / "dummy_dataset",
        relation_info=request.path.parent / "test_data" / "dummy_relation_info.json",
    )

    with pytest.raises(ValueError):
        callback = PubQuizCallback(
            trainer=trainer, evaluator=evaluator, dataset=probing_dataset, accumulate="non_existing"
        )

    callback = PubQuizCallback(trainer=trainer, evaluator=evaluator, dataset=probing_dataset, accumulate="domains")

    trainer.add_callback(callback)

    trainer.train()

    losses = []
    overall_probing_scores = []
    single_domain_probing_scores = []

    for log in logging_callback.log_data:
        logger.debug(log)
        if "loss" in log:
            losses.append(log["loss"])

        if "eval_dummy_dataset_score" in log:
            overall_probing_scores.append(log["eval_dummy_dataset_score"])

        # We should observe scores for the domains
        if "eval_dummy_dataset_d" in log:
            single_domain_probing_scores.append(log["eval_dummy_dataset_d"])

    assert losses[0] > losses[-1]

    logger.debug(overall_probing_scores)
    logger.debug(single_domain_probing_scores)

    assert len(overall_probing_scores) > 0
    assert overall_probing_scores[0] > overall_probing_scores[-1]  # Seeing the same sentences over and over can't help

    assert len(single_domain_probing_scores) > 0
    assert single_domain_probing_scores[0] > single_domain_probing_scores[-1]
