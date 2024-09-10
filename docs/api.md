# API Reference

You can use the API to call the evaluation from a python script.
For this, you need to load a dataset (see [Data Files](data_files.md) for how these should be structured)
and then execute the evaluation function using your desired configuration.

Example (compare with [`src/lm_pub_quiz/cli/evaluate_model.py`](https://github.com/lm-pub-quiz/lm-pub-quiz/tree/main/src/lm_pub_quiz/cli/evaluate_model.py)):


``` python
from lm_pub_quiz import Dataset, Evaluator

# Load dataset
dataset = Dataset.from_name("BEAR")

# Create Evaluator (and load model)
evaluator = Evaluator.from_model("distilbert-base-cased")

# Run evaluation
result = evaluator.evaluate_dataset(dataset)

# Save result object
result.save("outputs/my_results")
```


## Evaluator

::: lm_pub_quiz.Evaluator
    options:
        show_source: false
        show_root_heading: True
        heading_level: 3

::: lm_pub_quiz.MaskedLMEvaluator
    options:
        show_source: false
        show_root_heading: True
        heading_level: 3

::: lm_pub_quiz.CausalLMEvaluator
    options:
        show_source: false
        show_root_heading: True
        heading_level: 3


## Dataset Representation

There are two classes which are used to represent a dataset: `Relation` and `Dataset` (which is essentially a container for a number of relations).

::: lm_pub_quiz.Relation
    options:
        show_source: false
        show_root_heading: True
        heading_level: 3

::: lm_pub_quiz.Dataset
    options:
        show_source: false
        show_root_heading: True
        heading_level: 3


## Evaluation Result

Similar to the [dataset representation](#dataset-representation), the results are also represented in two classes `RelationResult` and the container `DatasetResults`.

::: lm_pub_quiz.RelationResult
    options:
        show_source: false
        show_root_heading: True
        heading_level: 3

::: lm_pub_quiz.DatasetResults
    options:
        show_source: false
        show_root_heading: True
        heading_level: 3


## Data Base Clasess

The dataset representations as well as the evaluation results are based on common base classes.



::: lm_pub_quiz.data.base.RelationBase
    options:
        show_source: fals
        show_root_heading: True
        heading_level: 3

::: lm_pub_quiz.data.base.DatasetBase
    options:
        show_source: false
        show_root_heading: True
        heading_level: 3
