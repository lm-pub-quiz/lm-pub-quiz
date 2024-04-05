import logging

import numpy as np
import pytest

from lm_pub_quiz import Dataset, DatasetResults, Evaluator, RelationResult
from lm_pub_quiz.data import NoInstanceTableError

log = logging.getLogger(__name__)


def test_evaluator_instantiations(distilbert):
    model, tokenizer = distilbert

    # Instantiate using model name
    Evaluator.from_model("distilbert-base-cased")

    # Instantiate from existing model
    Evaluator.from_model(model, tokenizer=tokenizer)


def test_reduction_functionality(distilbert):
    model, tokenizer = distilbert

    # Instantiate from existing model
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer)

    with pytest.raises(ValueError):
        evaluator.score_answers(
            template="The traveler lost the [Y].", answers=["souvenir", "bet"], reduction="non-existant"
        )

    with pytest.raises(ValueError):
        evaluator.evaluate_instance(
            template="The traveler lost the [Y]", answers=["souvenir", "bet"], reduction=None, print_ranking=True
        )


@pytest.mark.parametrize(
    "model_name, model_type",
    [
        ("distilbert-base-cased", "MLM"),
        ("xlm-roberta-large", "MLM"),
        ("biobert-base-cased", "MLM"),
        ("Llama-2-7b", "CLM"),
        ("opt-1.3b", "CLM"),
        ("gpt2-medium", "CLM"),
        ("bloom-1b7", "CLM"),
    ],
)
def test_model_type_inference(model_name, model_type):
    assert Evaluator._infer_type_from_name(model_name) == model_type


def test_incorrect_template(distilbert):
    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer)

    with pytest.raises(ValueError):
        evaluator.evaluate_instance(template="No object slot available", answers=["a", "b"], reduction="sum")


@pytest.mark.parametrize("lazy", [False, True])
def test_dataset_evaluation(distilbert, request, tmp_path, lazy):
    dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset")

    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer)

    results = evaluator.evaluate_dataset(dataset, batch_size=16, reduction="sum")
    assert isinstance(results, DatasetResults)
    assert all(isinstance(r, RelationResult) for r in results)

    results.save(tmp_path)
    log.debug("Contents of result directory:\n%s", "\n".join(str(p) for p in tmp_path.glob("*")))

    del results

    assert (tmp_path / "metadata_results.json").exists(), "Results metadata file expected but not found."
    assert (tmp_path / "example_1_results.jsonl").exists(), "Instance-file for example_1 expected but not found."
    assert (tmp_path / "example_2_results.jsonl").exists(), "Instance-file for example_2 expected but not found."

    results = DatasetResults.from_path(tmp_path, lazy=lazy)

    r: RelationResult
    for r in results:
        if r.relation_code in ("example_1", "example_2"):
            log.debug("Result for relation %s:\n%s", r.relation_code, str(r.instance_table))

            assert r.instance_table is not None, "Either the instance table was not created or not loaded correctly."
            a = r.instance_table
            b = r.instance_table

            if lazy:
                # table should not be saved in the relation object
                assert a is not b, f"Identity check failed {id(a)} - {id(b)}"
            else:
                assert a is b, f"Identity check failed {id(a)} - {id(b)}"

            # all examples should be predicted correctly
            for _, row in r.instance_table.iterrows():
                log.info(row)

                assert len(row["pll_scores"]) == 3
                assert row["answer_idx"] == np.argmax(row["pll_scores"])

            assert r.get_metric("accuracy") == 1.0

            metadata = r.get_metadata()

            assert metadata["dataset_name"] == "dummy_dataset"
            assert metadata["reduction"] == "sum"
            assert "distilbert" in metadata["model_name_or_path"]
            assert all(key in metadata.keys() for key in ["answer_space_ids", "answer_space_labels"])
            assert len(metadata["answer_space_ids"]) == len(metadata["answer_space_labels"])

    df = results.get_metrics(["accuracy", "precision_at_k"])
    assert df.loc["example_1", "accuracy"] == 1.0
    assert isinstance(df.loc["example_1", "precision_at_k"], list)

    for r in results:
        r.metric_values.clear()

    df = results.get_metrics(["accuracy", "precision_at_k"])
    assert df.loc["example_1", "accuracy"] == 1.0
    assert isinstance(df.loc["example_1", "precision_at_k"], list)
    assert df.loc["example_1", "precision_at_k"] == [1.0, 1.0, 1.0]

    df = results.get_metrics("accuracy")
    assert df.loc["example_1", "accuracy"] == 1.0


def test_dataset_evaluation_with_direct_scores(distilbert, request, tmp_path):
    log.debug("Lazy-loading the dataset")
    dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset")

    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer)

    log.debug("Starting evaluation")
    results = evaluator.evaluate_dataset(
        dataset, batch_size=16, reduction="sum", metric="accuracy", create_instance_table=False
    )

    log.debug("Assessing the results")

    assert isinstance(results, DatasetResults)
    assert all(isinstance(r, RelationResult) for r in results)

    results.save(tmp_path)

    log.debug("Contents of result directory:\n%s", "\n".join(str(p) for p in tmp_path.glob("*")))

    del results

    assert (tmp_path / "metadata_results.json").exists(), "Results metadata file expected but not found."

    with open(tmp_path / "metadata_results.json") as f:
        log.debug(f.read())

    assert not (tmp_path / "example_1_results.jsonl").exists(), "Instance-file for example_1 not expected but found."
    assert not (tmp_path / "example_2_results.jsonl").exists(), "Instance-file for example_2 not expected but found."

    results = DatasetResults.from_path(tmp_path, lazy=True)

    r: RelationResult
    for r in results:
        if r.relation_code in ("example_1", "example_2"):
            log.debug("Result for relation %s", r.relation_code)

            assert not r.has_instance_table

            with pytest.raises(NoInstanceTableError):
                _ = r.instance_table

            log.debug(r.metric_values)

            assert r.get_metric("accuracy") == 1.0

            assert r.metadata["dataset_name"] == "dummy_dataset"
            assert r.metadata["reduction"] == "sum"
            assert "distilbert" in r.metadata["model_name_or_path"]

    assert results.get_metrics(["accuracy"], accumulate_all=True)["accuracy"] >= 0.5


def test_dataset_evaluation_with_save_path(distilbert, request, tmp_path):
    dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset")

    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer)

    results = evaluator.evaluate_dataset(dataset, batch_size=16, reduction="sum", save_path=tmp_path)
    assert isinstance(results, DatasetResults)
    assert all(isinstance(r, RelationResult) for r in results)

    log.debug("Contents of result directory:\n%s", "\n".join(str(p) for p in tmp_path.glob("*")))

    assert (tmp_path / "metadata_results.json").exists(), "Results metadata file expected but not found."
    assert (tmp_path / "example_1_results.jsonl").exists(), "Instance-file for example_1 expected but not found."
    assert (tmp_path / "example_2_results.jsonl").exists(), "Instance-file for example_2 expected but not found."

    r: RelationResult
    for r in results:
        if r.relation_code in ("example_1", "example_2"):
            assert r._instance_table is None
            assert r.is_lazy

            log.debug("Result for relation %s:\n%s", r.relation_code, str(r.instance_table))

            a = r.instance_table
            b = r.instance_table

            assert a is not None
            assert b is not None

            # table should not be saved in the relation object
            assert a is not b, f"Identity check failed {id(a)} - {id(b)}"

            # all examples should be predicted correctly
            for _, row in a.iterrows():
                log.info(row)

                assert len(row["pll_scores"]) == 3
                assert row["answer_idx"] == np.argmax(row["pll_scores"])

            assert r.get_metric("accuracy") == 1.0

            assert r.metadata["dataset_name"] == "dummy_dataset"
            assert r.metadata["reduction"] == "sum"
            assert "distilbert" in r.metadata["model_name_or_path"]


def test_dataset_conditional_evaluation(distilbert, request, tmp_path):
    dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset")

    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer, conditional=True)

    results = evaluator.evaluate_dataset(dataset, batch_size=16, reduction="sum")
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
        if r.relation_code in ("example_1", "example_2"):
            instance_table = r.instance_table

            assert instance_table is not None

            log.debug("Result for relation %s:\n%s", r.relation_code, str(instance_table))
            # all examples should be predicted correctly
            for _, row in instance_table.iterrows():
                log.info(row)
                assert len(row["pll_scores"]) == 3
                assert row["answer_idx"] == np.argmax(row["pll_scores"])

            assert r.metadata["dataset_name"] == "dummy_dataset"
            assert r.metadata["reduction"] == "sum"
            assert "distilbert" in r.metadata["model_name_or_path"]


def test_token_scores_within_word_l2r(distilbert):
    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer, pll_metric="within_word_l2r")

    result, indices = zip(
        *evaluator.score_answers(template="The traveler lost the [Y].", answers=["bet", "souvenir"], reduction=None)
    )

    assert len(result) == 2

    assert len(result[0]) == 7  # The travel ##er lost the bet .
    assert len(result[1]) == 9  # The travel ##er lost the so ##uve ##nir .

    assert result[0][5][1] > sum(result[1][i][1] for i in range(5, 8))  # bet > so (pll instead of surprisal)

    assert indices[0]["answer"] == [5]
    assert indices[1]["answer"] == [5, 6, 7]


def test_token_scores_original(distilbert):
    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer, pll_metric="original")

    result, _ = zip(
        *evaluator.score_answers(template="The traveler lost the [Y].", answers=["bet", "souvenir"], reduction=None)
    )

    assert len(result) == 2

    assert len(result[0]) == 7  # The travel ##er lost the bet .
    assert len(result[1]) == 9  # The travel ##er lost the so ##uve ##nir .

    assert result[0][5][1] < sum(result[1][i][1] for i in range(5, 8))  # bet < so (pll instead of surprisal)


def test_conditional_score(distilbert):
    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer, conditional_score=True)

    result = evaluator.score_answers(
        template="The traveler lost the [Y].", answers=["bet", "souvenir"], reduction="sum"
    )

    assert -6 < result[0] < -5
    assert -13 < result[1] < -12


def test_automatic_tokenizer_loading(distilbert):
    model, _ = distilbert
    _ = Evaluator.from_model(model)

    # TODO: Add additional checks here
