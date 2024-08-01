"""
This script illustrates how to interact with the results object.

A result directory is required to run the script (e.g. as produced by `examples/direct_evaluation.py`).
"""

from lm_pub_quiz import DatasetResults

bear_results = DatasetResults.from_path("examples/gpt2_results")

print(bear_results.get_metrics(["accuracy", "support"], accumulate=True))


print(bear_results.get_metrics(["accuracy", "support"], accumulate="cardinality"))
