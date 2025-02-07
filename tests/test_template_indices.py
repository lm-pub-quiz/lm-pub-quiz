import logging

import numpy as np

from lm_pub_quiz import Dataset, DatasetResults, Evaluator

log = logging.getLogger(__name__)


def test_single_template_index(request, distilbert):
    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer)

    dataset = Dataset.from_path(
        request.path.parent / "test_data" / "dummy_dataset_multiple_templates",
    )

    results = evaluator.evaluate_dataset(
        dataset,
        template_index=0,
    )

    df = results.joined_instance_table()
    df["predicted_answer_idx"] = df["pll_scores"].apply(np.argmax)

    assert df["template_idx"].unique() == [0]

    assert (df["predicted_answer_idx"] == [0, 1, 2, 0, 0, 1, 1, 2, 2]).all()


def test_list_of_templates(request, distilbert):
    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer)

    dataset = Dataset.from_path(
        request.path.parent / "test_data" / "dummy_dataset_multiple_templates",
    )

    results = evaluator.evaluate_dataset(
        dataset,
        template_index=[0, 1],
    )

    df = results.joined_instance_table()

    assert df["template_idx"].unique() == [0, 1]


def test_all_templates(request, distilbert, tmp_path):
    model, tokenizer = distilbert
    evaluator = Evaluator.from_model(model, tokenizer=tokenizer)

    dataset = Dataset.from_path(
        request.path.parent / "test_data" / "dummy_dataset_multiple_templates",
    )

    results = evaluator.evaluate_dataset(
        dataset,
        template_index=None,
        save_path=tmp_path,
    )

    results = DatasetResults.from_path(tmp_path)
    df = results.joined_instance_table()

    assert df["template_idx"].unique() == [0, 1, 2]
