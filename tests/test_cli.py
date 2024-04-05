# TODO: This is potentially  to a changed path

import logging
import subprocess

from lm_pub_quiz.cli import evaluate_model_cli, rank_answers_cli, score_sentence_cli

log = logging.getLogger(__name__)


def test_help_texts():
    """Check the help text is ok, and the commands are available."""
    for script in ("evaluate_model", "rank_answers", "score_sentence"):
        result = subprocess.run(
            [str(script), "--help"], check=True, capture_output=True  # noqa: S603 (we can trust the script names)
        )
        stdout = result.stdout.decode("utf8")

        assert stdout.split("\n")[0].startswith(f"usage: {script}")

        for keyword in ("--model PATH", "--model.name_or_path", "--device"):
            assert keyword in stdout


def test_score_sentence(capsys):
    score_sentence_cli(
        [
            "--model.name",
            "distilbert-base-uncased",
            "--sentence",
            "The traveler lost the souvenir.",
        ]
    )

    stdout = capsys.readouterr().out
    assert len(stdout) > 0


def test_score_sentence_each_token(capsys):
    score_sentence_cli(
        [
            "--model.name",
            "distilbert-base-uncased",
            "--sentence",
            "The traveler lost the souvenir.",
            "--model.reduction",
            "none",
        ]
    )

    stdout = capsys.readouterr().out
    assert len(stdout) > 0


def test_rank_answers(capsys):
    rank_answers_cli(
        [
            "--model.name",
            "distilbert-base-uncased",
            "--template",
            "The traveler lost the [Y].",
            "--answers",
            "scalpel,souvenir",
        ]
    )

    stdout = capsys.readouterr().out
    assert len(stdout) > 0


def test_evaluate_model(capsys, tmp_path):
    evaluate_model_cli(
        [
            "--model.name",
            "distilbert-base-uncased",
            "--dataset_path",
            "tests/test_data/dummy_dataset",
            "--output_base_path",
            str(tmp_path),
        ]
    )

    _ = capsys.readouterr()

    path = tmp_path / "dummy_dataset" / "distilbert-base-uncased" / "within_word_l2r_sum"

    for p in reversed(path.parents):
        log.debug(", ".join(str(d) for d in p.parent.glob("*")))
        assert p.exists(), f"{p} not found."

    assert path.exists(), "Expected output path not found."

    log.debug("Contents of result directory:\n%s", "\n".join(str(p) for p in path.glob("*")))

    assert (path / "metadata_results.json").exists(), "Results metadata file expected but not found."
    assert (path / "example_1_results.jsonl").exists(), "Instance-file for example_1 expected but not found."
    assert (path / "example_2_results.jsonl").exists(), "Instance-file for example_2 expected but not found."


def test_evaluate_model_relation(capsys, tmp_path):
    evaluate_model_cli(
        [
            "--model.name",
            "distilbert-base-uncased",
            "--dataset_path",
            "tests/test_data/dummy_dataset",
            "--output_base_path",
            str(tmp_path),
            "--relation",
            "example_1",
        ]
    )

    _ = capsys.readouterr()

    path = tmp_path / "dummy_dataset" / "distilbert-base-uncased" / "within_word_l2r_sum"

    for p in reversed(path.parents):
        log.debug(", ".join(str(d) for d in p.parent.glob("*")))
        assert p.exists(), f"{p} not found."

    assert path.exists(), "Expected output path not found."

    log.debug("Contents of result directory:\n%s", "\n".join(str(p) for p in path.glob("*")))

    assert (path / "metadata_results.json").exists(), "Results metadata file expected but not found."
    assert (path / "example_1_results.jsonl").exists(), "Instance-file for example_1 expected but not found."
    assert not (
        path / "example_2_results.jsonl"
    ).exists(), "Instance-file for example_2 expected to not exist but found."


def test_evaluate_model_tyq(capsys, tmp_path):
    evaluate_model_cli(
        [
            "--model.name",
            "distilbert-base-uncased",
            "--dataset_path",
            "tests/test_data/dummy_dataset",
            "--model.reduction",
            "tyq",
            "--output_base_path",
            str(tmp_path),
        ]
    )

    _ = capsys.readouterr()

    path = tmp_path / "dummy_dataset" / "distilbert-base-uncased" / "tyq"

    for p in reversed(path.parents):
        log.debug(", ".join(str(d) for d in p.parent.glob("*")))
        assert p.exists(), f"{p} not found."

    assert path.exists(), "Expected output path not found."

    log.debug("Contents of result directory:\n%s", "\n".join(str(p) for p in path.glob("*")))

    assert (path / "metadata_results.json").exists(), "Results metadata file expected but not found."
    assert (path / "example_1_results.jsonl").exists(), "Instance-file for example_1 expected but not found."
    assert (path / "example_2_results.jsonl").exists(), "Instance-file for example_2 expected but not found."
