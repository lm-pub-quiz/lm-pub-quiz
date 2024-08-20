import logging

import pytest

from lm_pub_quiz.evaluators.templating_util import Templater

log = logging.getLogger(__name__)


def test_templater_without_placeholders():
    template = "Some sentence."

    templater = Templater()
    sentence, spans = templater.replace_placeholders(template=template, subject="Paris", answer="France")

    assert sentence == "Some sentence."
    assert spans["subject"] == []
    assert spans["answer"] == []
    assert len(spans) == 2


def test_templater_single_replacement():
    template = "[X] is the capital of [Y]."

    expected_text = "Paris is the capital of France."

    templater = Templater()
    text, spans = templater.replace_placeholders(template=template, subject="Paris", answer="France")

    assert text == expected_text
    assert spans["subject"] == [
        (0, 5),
    ]
    assert spans["answer"] == [
        (24, 30),
    ]
    assert len(spans) == 2


def test_templater_multiple_replacements():
    template = "[X] is the capital of [Y]. BERT. You could also: The capital of [Y] is [X]."

    templater = Templater()
    sentence, spans = templater.replace_placeholders(template=template, subject="paris", answer="France")

    assert sentence == "Paris is the capital of France. BERT. You could also: The capital of France is paris."
    assert spans["subject"] == [(0, 5), (79, 84)]
    assert spans["answer"] == [(24, 30), (69, 75)]
    assert len(spans) == 2


def test_templater_empty_subject():
    template = "A [X] sentence with an empty [Y]."

    templater = Templater()
    sentence, spans = templater.replace_placeholders(template=template, subject="  ", answer="replacement")

    assert sentence == "A    sentence with an empty replacement."
    assert spans["subject"] == [
        (2, 4),
    ]
    assert len(spans) == 2


@pytest.mark.parametrize("capitalize", [True, False])
def test_templater_subject_first(capitalize):
    template = "a [X] d e [Y] g."
    subject = "b  c "
    answer = "f"

    templater = Templater(capitalize=capitalize)
    text, spans = templater.replace_placeholders(template=template, subject=subject, answer=answer)

    assert len(spans) == 2

    assert (text == text.lower()) != capitalize
