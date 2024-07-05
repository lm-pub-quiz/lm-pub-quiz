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

    assert results.get_metrics(["accuracy"], accumulate=True)["accuracy"] == (1.0*3 + 0.5*6) / 9


def test_result_accumulation_with_relation_info(request):
    """Test whether the deprecated representation of the results can still be loaded."""

    results = DatasetResults.from_path(request.path.parent / "test_data" / "new_style_results_with_mistakes")

    results.update_relation_info({
        "example_1": {"domain": "a"},
        "example_2": {"domain": "b"}
    })

    df = results.get_metrics(["accuracy"], accumulate="domain")

    assert df.loc["a", "accuracy"] == 1.0
    assert df.loc["b", "accuracy"] == 0.5


def test_restults_with_multiple_tags(request):
    raise NotImplementedError()
