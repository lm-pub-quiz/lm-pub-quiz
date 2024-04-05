import pandas as pd

from lm_pub_quiz import Dataset, Relation


def test_relation_loading(request):
    relation: Relation = Relation.from_path(
        request.path.parent / "test_data" / "dummy_dataset", relation_code="example_1"
    )
    assert isinstance(relation, Relation)
    assert len(relation.templates) == 1
    assert len(relation.instance_table) == 3

    relation = Relation.from_path(request.path.parent / "test_data" / "dummy_dataset", relation_code="example_2")
    assert len(relation.instance_table) == 6
    assert len(relation.subsample(4)) == 4


def test_loading_dataset(request):
    dataset: Dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset")

    assert len(dataset) == 2
    assert str(dataset) == "Dataset(dummy_dataset: example_1, example_2)"


def test_lazy_dataset_loading(request):
    dataset: Dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset")

    assert dataset[0]._instance_table is None
    assert dataset[0].instance_table is not None
    assert isinstance(dataset[0].instance_table, pd.DataFrame)


def test_dataset_subset_filtering_lazy(request, tmp_path):
    dataset: Dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset", lazy=True)

    indices = {
        "example_1": [1, 2],
        "example_2": [0, 1, 4, 5],
    }

    subset = dataset.filter_subset(indices, save_path=tmp_path)

    assert subset.name == "dummy_dataset (subset)"

    assert subset[0]._instance_table is None
    assert subset.is_lazy

    assert len(subset[0]) == 2
    assert len(subset[1]) == 4

    assert list(subset[1].answer_space) == ["the lion", "the caterpillar"]

    for i in range(2):
        table = subset[i].instance_table
        assert (table["answer_idx"] == table["obj_id"].map(subset[i].answer_space.index.get_loc)).all()


def test_dataset_subset_filtering(request):
    dataset: Dataset = Dataset.from_path(request.path.parent / "test_data" / "dummy_dataset")

    indices = {
        "example_1": [1, 2],
        "example_2": [0, 1, 4, 5],
    }
    subset = dataset.filter_subset(indices, dataset_name="dummy_lite", keep_answer_space=True)

    assert subset.name == "dummy_lite"

    assert subset[0]._instance_table is not None
    assert not subset.is_lazy

    assert len(subset[0]) == 2
    assert len(subset[1]) == 4

    assert list(subset[1].answer_space) == ["the lion", "the kid", "the caterpillar"]
