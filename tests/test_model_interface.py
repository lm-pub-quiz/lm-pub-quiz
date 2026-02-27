import logging

import pytest

from lm_pub_quiz.model_interface.hf import CLMInterface
from lm_pub_quiz.model_interface.vllm import VLLMInterface

logger = logging.getLogger(__name__)

test_data = (
    ("The cat sleeps.", -25.459693431854248),
    ("The sun shines.", -20.245419025421143),
    ("The river flows.", -24.671634674072266),
)


def test_hf_clm(test_data=test_data):
    interface = CLMInterface.from_model("distilgpt2")

    statements, expected_scores = zip(*test_data, strict=True)

    scores = interface.score_statements(statements)

    for s, a, b in zip(statements, scores, expected_scores, strict=True):
        assert a == pytest.approx(b), s


@pytest.mark.skip(reason="Requires correct environment.")
def test_vllm(test_data=test_data):
    model = "distilgpt2"
    logger.debug("Creating VLLM Interface for model %s", model)
    interface = VLLMInterface.from_model(model, dtype="float32")

    logger.debug("Scoring statememts")
    statements, expected_scores = zip(*test_data, strict=True)

    scores = list(interface.score_statements(statements))

    logger.debug("Comparing results (low precision)")
    for s, a, b in zip(statements, scores, expected_scores, strict=True):
        assert a == pytest.approx(b, abs=1e-2), f"{s}: {a} vs. {b}"

    logger.info("Results approx. ok (1e-2 precision)")

    logger.debug("Comparing results")
    for s, a, b in zip(statements, scores, expected_scores, strict=True):
        assert a == pytest.approx(b), f"{s}: {a} vs. {b}"


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    test_vllm()
    logging.getLogger("__main__").info("Done")
