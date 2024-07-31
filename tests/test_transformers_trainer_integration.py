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


class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.log_data = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.log_data.append(logs)


def test_callback(request, tmp_path):
    probing_dataset = Dataset.from_path(
        request.path.parent / "test_data" / "dummy_dataset",
        relation_info=request.path.parent / "test_data" / "dummy_relation_info.json",
    )

    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]], truncation=True, padding="max_length", max_length=512)

    mlm_eval_dataset = HFDataset.from_dict(
        {
            "text": ["I will not waste chalk.", "I will not skateboard in the halls."]
            + ["I will not cut corners."] * 6,
        }
    ).map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    mlm_train_dataset = HFDataset.from_dict({"text": ["I will not bury the new kid."] * 10}).map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        tmp_path,
        do_train=True,
        num_train_epochs=1,
        logging_first_step=True,
        logging_steps=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to="none",
    )

    logging_callback = LoggingCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mlm_train_dataset,
        eval_dataset=mlm_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[logging_callback],
    )

    evaluator = Evaluator.from_model(
        model,
        tokenizer=tokenizer,
        model_type="CLM",
        batch_size=1,
        dataset=probing_dataset,
        template_index=0,
    )

    callback = PubQuizCallback(trainer=trainer, evaluator=evaluator)

    trainer.add_callback(callback)

    trainer.train()

    print(logging_callback.log_data)

    losses = []
    bear_scores = []

    for log in logging_callback.log_data:
        if "loss" in log:
            losses.append(log["loss"])

        if "bear_score" in log:
            bear_scores.append(bear_scores)

    assert losses[0] > losses[-1]

    # TODO: check that the log_data actually contains the relevant scores
    assert len(bear_scores) > 0
    assert bear_scores[0] > bear_scores[-1]  # Seeing the same sentences over and over can't help
