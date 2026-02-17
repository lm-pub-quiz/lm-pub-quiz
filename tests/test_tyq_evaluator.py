import logging

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
            [-8.248769, -8.074352, -7.663999],
            [-9.166306, -5.309185, -9.256723],
            [-9.397379, -7.905240, -7.342500],
        ],
        [
            [-5.399512, -7.076719, -7.062878],
            [-5.234767, -5.896080, -6.817871],
            [-6.116759, -5.320151, -7.580444],
            [-6.166567, -6.215422, -7.152047],
            [-6.133464, -6.806790, -6.393774],
            [-6.159425, -6.892731, -6.757021],
        ],
    ]

    r: RelationResult
    for r, exp_scores in zip(results, expected_scores):
        instance_table = r.instance_table

        assert instance_table is not None

        for i, row in instance_table.iterrows():
            for a, b in zip(row["pll_scores"], exp_scores[i]):
                assert a == pytest.approx(b), f"Actual {row['pll_scores']!s} vs expected: {exp_scores[i]!s}"

        if r.relation_code in ("example_1"):
            log.debug("Result for relation %s:\n%s", r.relation_code, str(instance_table))

            assert r.get_metadata("dataset_name") == "dummy_dataset"
            assert "distilbert" in r.get_metadata("model_name_or_path")


def test_reduction_functionality(distilbert):
    model, tokenizer = distilbert

    # Instantiate from existing model

    model_interface = TyQModelInterface.from_model(model, tokenizer=tokenizer)

    evaluator = Evaluator(model_interface=model_interface)

    _ = evaluator.evaluate_item(template="The traveler lost the [Y]", answers=["souvenir", "bet"])
