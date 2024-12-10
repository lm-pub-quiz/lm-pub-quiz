from lm_pub_quiz import Dataset, Evaluator

result_save_path = "results"
model_name = "bigscience/bloom-7b1"

# Load the BEAR dataset from its specific location
dataset = Dataset.from_name("BEAR")

# Run the BEAR evaluator and save the results
evaluator = Evaluator.from_model(model_name, model_type="CLM", model_kw={"device_map": "auto"}, device="cuda")
results = evaluator.evaluate_dataset(dataset, save_path=result_save_path, batch_size=32)
