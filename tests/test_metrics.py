import logging

from lm_pub_quiz import DatasetResults

log = logging.getLogger(__name__)


def test_result_accumulation(request):
    """Test whether the deprecated representation of the results can still be loaded."""

    results = DatasetResults.from_path(request.path.parent / "test_data" / "new_style_results_with_mistakes")

    df = results.get_metrics(["accuracy"], accumulate=False)

    # todo add an example where the accuracies are different

    assert df.loc["example_1", "accuracy"] == 1.0
    assert df.loc["example_2", "accuracy"] == 0.5

    assert results.get_metrics(["accuracy"], accumulate=True)["accuracy"] == (1.0 * 3 + 0.5 * 6) / 9


def test_result_accumulation_with_relation_info(request):
    """Test whether the deprecated representation of the results can still be loaded."""

    results = DatasetResults.from_path(request.path.parent / "test_data" / "new_style_results_with_mistakes")

    results.update_relation_info({"example_1": {"domain": "a"}, "example_2": {"domain": "b"}})

    df = results.get_metrics(["accuracy"], accumulate="domain")

    assert df.loc["a", "accuracy"] == 1.0
    assert df.loc["a", "support"] == 3.0
    assert df.loc["b", "accuracy"] == 0.5
    assert df.loc["b", "support"] == 6.0


def test_results_with_multiple_tags_without_dividied_support(request):
    results = DatasetResults.from_path(request.path.parent / "test_data" / "new_style_results_with_mistakes")

    results.update_relation_info({"example_1": {"domain": ("a", "b", "c")}, "example_2": {"domain": ("b", "d")}})

    df = results.get_metrics(["accuracy"], accumulate="domain", explode=True, divide_support=False)

    assert df.loc["a", "accuracy"] == 1.0
    assert df.loc["a", "support"] == 3
    assert df.loc["b", "accuracy"] == (3 * 1.0 + 0.5 * 6) / 9
    assert df.loc["b", "support"] == 9
    assert df.loc["c", "accuracy"] == 1.0
    assert df.loc["c", "support"] == 3
    assert df.loc["d", "accuracy"] == 0.5
    assert df.loc["d", "support"] == 6


def test_results_with_multiple_tags_divided_support(request):
    results = DatasetResults.from_path(request.path.parent / "test_data" / "new_style_results_with_mistakes")

    results.update_relation_info({"example_1": {"domain": ("a", "b", "c")}, "example_2": {"domain": ("b", "d")}})

    df = results.get_metrics(["accuracy"], accumulate="domain", explode=True)

    assert df.loc["a", "accuracy"] == 1.0
    assert df.loc["a", "support"] == 1.0
    assert df.loc["b", "accuracy"] == (1 * 1.0 + 0.5 * 3) / 4
    assert df.loc["b", "support"] == 4
    assert df.loc["c", "accuracy"] == 1.0
    assert df.loc["c", "support"] == 1
    assert df.loc["d", "accuracy"] == 0.5
    assert df.loc["d", "support"] == 3


def test_result_accumulation_with_relation_info_from_path(request):
    """Test whether the deprecated representation of the results can still be loaded."""

    results = DatasetResults.from_path(
        request.path.parent / "test_data" / "new_style_results_with_mistakes",
        relation_info=request.path.parent / "test_data" / "dummy_relation_info.json",
    )
    df = results.get_metrics(["accuracy"], accumulate="domain", explode=True)

    assert df.loc["a", "accuracy"] == 1.0
    assert df.loc["a", "support"] == 1.0
    assert df.loc["b", "accuracy"] == (1 * 1.0 + 0.5 * 3) / 4
    assert df.loc["b", "support"] == 4
    assert df.loc["c", "accuracy"] == 1.0
    assert df.loc["c", "support"] == 1
    assert df.loc["d", "accuracy"] == 0.5
    assert df.loc["d", "support"] == 3
