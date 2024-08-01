# Integration with Huggingface's Trainer
The lm_pub_quiz package allows for easy integration of the BEAR probe with the Huggingface Trainer. This way models can be continuously evaluated throughout training.

## LM Pub Quiz Callback
Here is an example of how to set up the Trainer Callback needed for integration.

```python
# import the PubQuiz Trainer Callback class
from lm_pub_quiz.integrations.transformers_trainer import PubQuizCallback

# set up the trainer as you usually would
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# load the BEAR dataset
bear_dataset = Dataset.from_path("../BEAR")

# create evaluator and Trainner callback
evaluator = Evaluator.from_model(
    model=model,
    tokenizer=tokenizer,
)
pub_quiz_callback = PubQuizCallback(
    trainer=trainer,
    evaluator=evaluator,
    dataset=bear_dataset,
    save_path="./bear_results",
)

# add the PubQuiz callback to the Trainer
trainer.add_callback(pub_quiz_callback)

# run training
trainer.train()
```
Setting up the trainer prior to the callback is needed for the callback to have access to the logging functionality of the trainer. This way BEAR evaluation results are automatically logged by the trainer. And thus for example also included in a Tensorboard or Weights and Biases report.

## Optional Parameters
The LM Pub Quiz Callback performs the BEAR probe automatically, whenever the evaluation strategy of the Trainer calls for it. It's behaviour can be further customized with a number of optional parameters:

| Argument    | Description                                                                     |
|:------------|:--------------------------------------------------------------------------------|
| `save_path` | If other than `None`, full BEAR evaluation results are saved to this directory. |
| `metrics`   | one of [None, "overall", "domains", "cardinality"]                              |
| `template`  | either specify the template index a pass a list of template indices             |

## Complete Example
After the `training_run()` was completed you can view the reported metrics by calling `tensorboard --logdir logs/`. And inspect the full BEAR results saved at `<PATH TO BEAR RESULTS SAVE DIR>`.
```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from lm_pub_quiz.integrations.transformers_trainer import PubQuizCallback
from lm_pub_quiz.data import Dataset
from lm_pub_quiz.evaluator import Evaluator


def training_run(
        device="cpu",
        dataset_reduction_factor=0.002,
):
    # Load reduced dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    select_train_indices = list(range(int(len(dataset['train']) * dataset_reduction_factor)))
    select_validation_indices = list(range(int(len(dataset['validation']) * dataset_reduction_factor)))
    dataset['train'] = dataset['train'].select(select_train_indices)
    dataset['validation'] = dataset['validation'].select(select_validation_indices)

    # load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", truncation=True, max_length=512)

    # do some minimal data preparation
    def filter_text(example):
        text = example['text']
        if len(text) < 100:
            return False
        if text[0] == '=' and text[-1] == '=':
            return False
        return True

    dataset = dataset.filter(filter_text)

    # tokenize data
    def tokenize(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_train_ds = dataset['train'].map(tokenize, batched=True, remove_columns=['text'])
    tokenized_eval_ds = dataset['validation'].map(tokenize, batched=True, remove_columns=['text'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    # set up trainer with reporting to tensorboard
    model_save_dir = './models/run_01'
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        run_name="run_01",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=5e-01,
        lr_scheduler_type='constant',
        eval_steps=5,
        eval_strategy="steps",
        logging_steps=5,
        logging_dir='logs',
        report_to='tensorboard',
        push_to_hub=False,
        save_strategy="no",
        include_num_input_tokens_seen=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # set up BEAR integration callback
    dataset_path = "<PATH TO BEAR DATASET>"
    bear_dataset = Dataset.from_path(dataset_path, relation_info="/home/seb/test_lm_pub_quiz/relation_info.json")
    bear_dataset = bear_dataset.filter_subset({"P6": list(range(5)), "P30": list(range(10)), "P103": list(range(5)), "P175": list(range(10))})
    evaluator = Evaluator.from_model(
        model=model,
        tokenizer=tokenizer,
        model_type="MLM",
        device="cpu",
    )
    pub_quiz_callback = PubQuizCallback(
        trainer=trainer,
        evaluator=evaluator,
        dataset=bear_dataset,
        save_path="/home/seb/test_lm_pub_quiz/results",
        metrics="domains",
    )
    trainer.add_callback(pub_quiz_callback)

    # run training
    trainer.train()
    trainer.save_model()

training_run()
```
