import logging

from lm_pub_quiz import Dataset, DatasetResults, RelationResult
from lm_pub_quiz.tyq_evaluator import TyQEvaluator

log = logging.getLogger(__name__)


def test_tyq_evaluator(distilbert, request, tmp_path):
    dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset")

    model, tokenizer = distilbert
    evaluator = TyQEvaluator.from_model(model, tokenizer=tokenizer)

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

    r: RelationResult
    for r in results:
        if r.relation_code in ("example_1"):
            instance_table = r.instance_table

            log.debug("Result for relation %s:\n%s", r.relation_code, str(instance_table))

            assert instance_table is not None

            # all examples should be predicted correctly
            for _, row in instance_table.iterrows():
                assert len(row["pll_scores"]) == 3
                assert row["answer_idx"] == row["pll_scores"].index(max(row["pll_scores"]))

            assert r.metadata["dataset_name"] == "dummy_dataset"
            assert "distilbert" in r.metadata["model_name_or_path"]


def test_reduction_functionality(distilbert):
    model, tokenizer = distilbert

    # Instantiate from existing model
    evaluator = TyQEvaluator.from_model(model, tokenizer=tokenizer)

    _ = evaluator.evaluate_instance(template="The traveler lost the [Y]", answers=["souvenir", "bet"])
