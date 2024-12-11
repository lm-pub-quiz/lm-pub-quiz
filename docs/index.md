# Getting started

This library implements a knowledge probing approach which uses LM's inherent ability to estimate the log-likelihood of any given textual statement.
For more information visit the [LM Pub Quiz website](https://lm-pub-quiz.github.io/).

![Illustration of how LM Pub Quiz evaluates LMs: Answers are ranked by the (pseudo) log-likelihoods of the textual statements derived from all of the answer options.](https://lm-pub-quiz.github.io/media/bear_evaluation_final.svg)

/// caption
Illustration of how LM Pub Quiz evaluates LMs: Answers are ranked by the (pseudo) log-likelihoods of the textual statements derived from all of the answer options.
///


The following sections give a quick overview how to calculate the BEAR-score for a given model. For a more detailed look into the results, please take a look at the [example workflow](example.md).


## Installing the Package

To install the package from PyPI, simply run:


```shell
pip install lm-pub-quiz
```

!!! Note

    For alternative setups (esp. for contributing to the library), see the [development section](development.md).




## Evaluating a Model

Models can be loaded and evaluated using the `Evaluator` class. First, create an evaluator for the model, then run `evaluate_dataset` with the loaded dataset.


```python
from lm_pub_quiz import Dataset, Evaluator

# Load the dataset
dataset = Dataset.from_name("BEAR")

# Load the model
evaluator = Evaluator.from_model(
    "gpt2",
    model_type="CLM",
)
# Run the evaluation and save the
results = evaluator.evaluate_dataset(
    dataset,
    template_index=0,
    save_path="gpt2_results",
    batch_size=32,
)
```

## Assessing the Results

To load the results and compute the overall accuracy, you can use the following lines of code:

```python
from lm_pub_quiz import DatasetResults

results = DatasetResults.from_path("gpt2_results")

print(results.get_metrics("accuracy"))
```
