import logging

import numpy as np
import pytest

from lm_pub_quiz import Dataset, DatasetResults, Evaluator, RelationResult
from lm_pub_quiz.model_interface.hf.tyq import TyQModelInterface

log = logging.getLogger(__name__)


def test_tyq_evaluator(distilbert, request, tmp_path):
    dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset")

    model, tokenizer = distilbert

    model_interface = TyQModelInterface.from_model(model, tokenizer=tokenizer)

    evaluator = Evaluator(model_interface=model_interface)

    results = evaluator.evaluate_dataset(dataset)
    assert isinstance(results, DatasetResults)
    assert all(isinstance(r, RelationResult) for r in results)

    results.save(tmp_path)
    log.debug("Contents of result directory:\n%s", "\n".join(str(p) for p in tmp_path.glob("*")))

    del results

    assert (tmp_path / "metadata_results.json").exists(), "Results metadata file expected but not found."
    assert (tmp_path / "example_1_results.jsonl").exists(), "Instance-file for example_1 expected but not found."
    assert (tmp_path / "example_2_results.jsonl").exists(), "Instance-file for example_2 expected but not found."

    results = DatasetResults.from_path(tmp_path)

    expected_scores = [
        [
            [-10.273433685302734, -10.768592834472656, -10.525314331054688],
            [-10.644917488098145, -9.782225608825684, -10.903377532958984],
            [-10.702713012695312, -10.465919494628906, -10.014009475708008],
        ],
        [
            [-9.771785736083984, -10.131648063659668, -9.680573463439941],
            [-9.440163612365723, -9.856391906738281, -9.055619239807129],
            [-10.2735595703125, -9.828554153442383, -10.340668678283691],
            [-10.120119094848633, -10.133556365966797, -10.010688781738281],
            [-9.742051124572754, -10.160009384155273, -8.927111625671387],
            [-10.028517723083496, -10.336956024169922, -8.990185737609863],
        ],
    ]

    r: RelationResult
    for r, exp_scores in zip(results, expected_scores):
        instance_table = r.instance_table

        for i, row in instance_table.iterrows():
            assert all(a == pytest.approx(b) for a, b in zip(row["pll_scores"], exp_scores[i])), (
                f"Expected {row['pll_scores']!s} vs found: {exp_scores[i]!s}"
            )

        if r.relation_code in ("example_1"):
            log.debug("Result for relation %s:\n%s", r.relation_code, str(instance_table))

            assert instance_table is not None

            # all examples should be predicted correctly
            for _, row in instance_table.iterrows():
                assert len(row["pll_scores"]) == 3
                assert row["answer_idx"] == np.argmax(row["pll_scores"])

            assert r.get_metadata("dataset_name") == "dummy_dataset"
            assert "distilbert" in r.get_metadata("model_name_or_path")


def test_reduction_functionality(distilbert):
    model, tokenizer = distilbert

    # Instantiate from existing model

    model_interface = TyQModelInterface.from_model(model, tokenizer=tokenizer)

    evaluator = Evaluator(model_interface=model_interface)

    _ = evaluator.evaluate_item(template="The traveler lost the [Y]", answers=["souvenir", "bet"])
