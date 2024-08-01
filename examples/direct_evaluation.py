"""This script illustrate how to load the dataset and perform an evaluation on a model."""

from lm_pub_quiz import Dataset, Evaluator

# Load the dataset
dataset = Dataset.from_name("BEAR")

# Load the model
evaluator = Evaluator.from_model(
    "gpt2",
    model_type="CLM",
)
# Run the evaluation and save the
evaluator.evaluate_dataset(
    dataset,
    template_index=0,
    save_path="gp2_results",
    batch_size=32,
)
