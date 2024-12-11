# API Reference

You can use the API to call the evaluation from a python script.
For this, you need to load a dataset (see [Data Files](data_files.md) for how these should be structured)
and then execute the evaluation function using your desired configuration.


``` python title="Example"
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












