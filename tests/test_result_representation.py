import logging

import numpy as np
import pytest

from lm_pub_quiz import DatasetResults, RelationResult

log = logging.getLogger(__name__)


@pytest.mark.parametrize("lazy", [False, True])
def test_new_style_results_loading(request, lazy):
    """Test whether the deprecated representation of the results can still be loaded."""

    results = DatasetResults.from_path(request.path.parent / "test_data" / "new_style_results", lazy=lazy)

    assert isinstance(results, DatasetResults)
    assert len(results) == 2

    r: RelationResult
    for r in results:
        assert isinstance(r, RelationResult)

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

            assert r.metadata["dataset_name"] == "dummy_dataset"
            assert r.metadata["reduction"] == "sum"

            assert "distilbert" in r.metadata["model_name_or_path"]

    df = results.get_metrics(["accuracy", "precision_at_k"])

    assert df.loc["example_1", "accuracy"] == 1.0
    assert isinstance(df.loc["example_1", "precision_at_k"], list)

    for r in results:
        del r.metadata["metrics"]

    df = results.get_metrics(["accuracy", "precision_at_k"])

    assert df.loc["example_1", "accuracy"] == 1.0
    assert isinstance(df.loc["example_1", "precision_at_k"], list)
    assert df.loc["example_1", "precision_at_k"] == [1.0, 1.0, 1.0]

    df = results.get_metrics("accuracy")
    assert df.loc["example_1", "accuracy"] == 1.0


@pytest.mark.parametrize("lazy", [False, True])
def test_old_style_results_loading(request, lazy):
    """Test whether the deprecated representation of the results can still be loaded."""

    results = DatasetResults.from_path(request.path.parent / "test_data" / "old_style_results_reduced", lazy=lazy)

    assert isinstance(results, DatasetResults)
    assert len(results) == 2

    r: RelationResult
    for r in results:
        assert isinstance(r, RelationResult)

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

            assert r.metadata["dataset_name"] == "dummy_dataset"
            assert r.metadata["reduction"] == "sum"

            assert "distilbert" in r.metadata["model_name_or_path"]

    df = results.get_metrics(["accuracy", "precision_at_k"])

    assert df.loc["example_1", "accuracy"] == 1.0
    assert isinstance(df.loc["example_1", "precision_at_k"], list)

    for r in results:
        del r.metadata["metrics"]

    df = results.get_metrics(["accuracy", "precision_at_k"])

    assert df.loc["example_1", "accuracy"] == 1.0
    assert isinstance(df.loc["example_1", "precision_at_k"], list)
    assert df.loc["example_1", "precision_at_k"] == [1.0, 1.0, 1.0]

    df = results.get_metrics("accuracy")
    assert df.loc["example_1", "accuracy"] == 1.0


def test_old_style_results_non_reduced(request):
    """Test whether the deprecated representation of the results can still be loaded."""

    results_with_indices = DatasetResults.from_path(
        request.path.parent / "test_data" / "old_style_results_with_indices"
    )

    assert isinstance(results_with_indices, DatasetResults)
    assert len(results_with_indices) == 2

    rel = results_with_indices[0]

    instance_table = rel.instance_table

    assert len(instance_table["template_indices"].iloc[0]) == len(rel.answer_space)

    for role in ("template", "sub", "obj"):
        log.debug(role)
        assert isinstance(instance_table[f"{role}_indices"].iloc[0][0], list)
        assert isinstance(instance_table[f"{role}_indices"].iloc[0][1], list)

    assert isinstance(instance_table["tokens"].iloc[0][0], list)
    assert isinstance(instance_table["tokens"].iloc[0][1], list)

    # The indices together should add up to the total length of the predicted statement
    assert sum(len(instance_table[f"{role}_indices"].iloc[0][1]) for role in ("template", "sub", "obj")) == len(
        instance_table["tokens"].iloc[0][1]
    )

    # Manually set indices should be preserved
    assert instance_table["template_indices"].iloc[0][0] == list(range(10))
    assert instance_table["sub_indices"].iloc[0][0] == list(range(10, 20))
    assert instance_table["obj_indices"].iloc[0][0] == list(range(20, 30))


@pytest.mark.parametrize("lazy_result", [False, True])
def test_result_subset_same_answer_space(request, tmp_path, lazy_result):
    """Test the creation of a subset of the results."""

    results = DatasetResults.from_path(request.path.parent / "test_data" / "old_style_results_reduced", lazy=True)

    indices = {
        "example_1": [1, 2],
        "example_2": [0, 1, 4, 5],
    }

    if lazy_result:
        results.filter_subset(indices, dataset_name="dummy_subset", keep_answer_space=True, save_path=tmp_path)

        subset = DatasetResults.from_path(tmp_path)

    else:
        subset = results.filter_subset(indices, dataset_name="dummy_subset", keep_answer_space=True, save_path=None)

    r: RelationResult
    for r in subset:
        assert isinstance(r, RelationResult)

        if r.relation_code in ("example_1", "example_2"):
            log.debug("Result for relation %s:\n%s", r.relation_code, str(r.instance_table))

            assert r.instance_table is not None, "Either the instance table was not created or not loaded correctly."
            a = r.instance_table
            b = r.instance_table

            if lazy_result:
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

            assert r.metadata["dataset_name"] == "dummy_subset"
            assert r.metadata["reduction"] == "sum"

            assert "distilbert" in r.metadata["model_name_or_path"]

    df = results.get_metrics(["accuracy", "precision_at_k"])

    assert df.loc["example_1", "accuracy"] == 1.0
    assert isinstance(df.loc["example_1", "precision_at_k"], list)

    for r in results:
        del r.metadata["metrics"]

    df = results.get_metrics(["accuracy", "precision_at_k"])

    assert df.loc["example_1", "accuracy"] == 1.0
    assert isinstance(df.loc["example_1", "precision_at_k"], list)
    assert df.loc["example_1", "precision_at_k"] == [1.0, 1.0, 1.0]

    df = results.get_metrics("accuracy")
    assert df.loc["example_1", "accuracy"] == 1.0


@pytest.mark.parametrize("lazy_result", [False, True])
def test_result_subset_smaller_answer_space(request, tmp_path, lazy_result):
    """Test the creation of a subset of the results."""

    results = DatasetResults.from_path(request.path.parent / "test_data" / "old_style_results_with_indices", lazy=True)

    indices = {
        "example_1": [1, 2],
        "example_2": [0, 1, 4, 5],
    }

    subset = results.filter_subset(indices, save_path=tmp_path if lazy_result else None)

    r: RelationResult
    for r in subset:
        assert isinstance(r, RelationResult)

        if r.relation_code in ("example_1", "example_2"):
            log.debug("Result for relation %s:\n%s", r.relation_code, str(r.instance_table))

            assert r.instance_table is not None, "Either the instance table was not created or not loaded correctly."
            a = r.instance_table
            b = r.instance_table

            if lazy_result:
                # table should not be saved in the relation object
                assert a is not b, f"Identity check failed {id(a)} - {id(b)}"
            else:
                assert a is b, f"Identity check failed {id(a)} - {id(b)}"

            # all examples should be predicted correctly
            for _, row in r.instance_table.iterrows():
                log.info(row)

                columns = ["tokens", "pll_scores", "sub_indices", "obj_indices", "template_indices"]
                for c in columns:
                    assert len(row[c]) == 2

                assert row["answer_idx"] == np.argmax(np.array([sum(v) for v in row["pll_scores"]]))

            assert r.reduced().get_metric("accuracy") == 1.0
            with pytest.warns(UserWarning):
                assert r.get_metric("accuracy") == 1.0

            assert r.metadata["dataset_name"] == "dummy_dataset"
            assert r.metadata["reduction"] is None

            assert "distilbert" in r.metadata["model_name_or_path"]

    with pytest.warns(UserWarning):

        df = subset.get_metrics(["accuracy", "precision_at_k"])

        assert df.loc["example_1", "accuracy"] == 1.0
        assert isinstance(df.loc["example_1", "precision_at_k"], list)
        assert df.loc["example_1", "precision_at_k"] == [1.0, 1.0]

        df = results.get_metrics("accuracy")
        assert df.loc["example_1", "accuracy"] == 1.0
