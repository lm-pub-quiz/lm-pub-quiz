"""
This script illustrates how to interact with the results object.

A result directory is required to run the script (e.g. as produced by `examples/direct_evaluation.py`).
"""

from lm_pub_quiz import DatasetResults

bear_results = DatasetResults.from_path("examples/gpt2_results")

print("=== Overall score ===")
print(bear_results.get_metrics(["accuracy", "support"], accumulate=True))
print()

print("=== Score by cardinality ===")
print(bear_results.get_metrics(["accuracy", "support"], accumulate="cardinality"))
print()

# Load the results with the additional relation information
bear_results = DatasetResults.from_path("examples/gpt2_results", relation_info="../BEAR/relation_info.json")

print("=== Score by domain ===")
# Since each relation can have multiple domains, we need to use the explode argument
print(bear_results.get_metrics(["accuracy", "support"], accumulate="domains", explode=True))
