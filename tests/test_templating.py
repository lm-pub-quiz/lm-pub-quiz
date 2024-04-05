import logging

import pytest
from transformers import AutoTokenizer

from lm_pub_quiz import Evaluator
from lm_pub_quiz.templating import Templater

log = logging.getLogger(__name__)


@pytest.mark.parametrize("tokenizer_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_templater_without_placeholders(tokenizer_name):
    template = "Some sentence."

    templater = Templater()
    sentence, spans = templater.replace_placeholders(template=template, subject="Paris", answer="France")

    assert sentence == "Some sentence."
    assert spans["subject"] == []
    assert spans["answer"] == []
    assert len(spans) == 2

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    _, indices = templater.tokenize_with_span_dict(tokenizer=tokenizer, text=sentence, spans=spans)

    assert len(indices) == 3
    assert indices["subject"] == []
    assert indices["answer"] == []


@pytest.mark.parametrize("tokenizer_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_templater_single_replacement(tokenizer_name):
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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokens, indices = templater.tokenize_with_span_dict(
        tokenizer=tokenizer, text=text, spans=spans, include_special_tokens=(tokenizer_name != "distilbert-base-cased")
    )

    assert len(indices) == 3

    log.info(dict(enumerate(tokenizer.convert_ids_to_tokens(tokens["input_ids"]))))
    log.info(indices)

    if tokenizer_name == "gpt2":
        assert indices["subject"] == [0]
        assert indices["answer"] == [5]
    elif tokenizer_name == "distilbert-base-cased":
        assert indices["subject"] == [0]
        assert indices["answer"] == [5]

    elif tokenizer_name == "facebook/opt-125m":
        assert indices["subject"] == [1]
        assert indices["answer"] == [6]
    else:
        msg = f"No check defined for {tokenizer_name}."
        raise RuntimeError(msg)


@pytest.mark.parametrize("tokenizer_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_templater_multiple_replacements(tokenizer_name):
    template = "[X] is the capital of [Y]. BERT. You could also: The capital of [Y] is [X]."

    templater = Templater()
    sentence, spans = templater.replace_placeholders(template=template, subject="paris", answer="France")

    assert sentence == "Paris is the capital of France. BERT. You could also: The capital of France is paris."
    assert spans["subject"] == [(0, 5), (79, 84)]
    assert spans["answer"] == [(24, 30), (69, 75)]
    assert len(spans) == 2

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens, indices = templater.tokenize_with_span_dict(
        tokenizer=tokenizer,
        text=sentence,
        spans=spans,
        include_special_tokens=(tokenizer_name != "distilbert-base-cased"),
    )

    assert len(indices) == 3

    log.info(dict(enumerate(tokenizer.convert_ids_to_tokens(tokens["input_ids"]))))
    log.info(indices)

    if tokenizer_name == "gpt2":
        assert indices["subject"] == [0, 19, 20]  # Upper case is single token, lower case is split in two
        assert indices["answer"] == [5, 17]
    elif tokenizer_name == "distilbert-base-cased":
        assert indices["subject"] == [0, 20, 21]
        assert indices["answer"] == [5, 18]
    elif tokenizer_name == "facebook/opt-125m":
        assert indices["subject"] == [1, 20, 21]
        assert indices["answer"] == [6, 18]
    else:
        msg = f"No check defined for {tokenizer_name}."
        raise RuntimeError(msg)


@pytest.mark.parametrize("tokenizer_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_templater_empty_subject(tokenizer_name):
    template = "A [X] sentence with an empty [Y]."

    templater = Templater()
    sentence, spans = templater.replace_placeholders(template=template, subject="  ", answer="replacement")

    assert sentence == "A    sentence with an empty replacement."
    assert spans["subject"] == [
        (2, 4),
    ]
    assert len(spans) == 2

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokens, indices = templater.tokenize_with_span_dict(
        tokenizer=tokenizer,
        text=sentence,
        spans=spans,
        include_special_tokens=(tokenizer_name != "distilbert-base-cased"),
    )

    assert len(indices) == 3

    log.info(dict(enumerate(tokenizer.convert_ids_to_tokens(tokens["input_ids"]))))
    log.info(indices)

    if tokenizer_name == "gpt2":
        assert indices["subject"] == [2, 3]  # extra spaces are encoded
        assert indices["answer"] == [8]

    elif tokenizer_name == "distilbert-base-cased":
        assert indices["subject"] == []  # space are dropped
        assert indices["answer"] == [5]

    elif tokenizer_name == "facebook/opt-125m":
        assert indices["subject"] == [3, 4]
        assert indices["answer"] == [9]

    else:
        msg = f"No check defined for {tokenizer_name}."
        raise RuntimeError(msg)


@pytest.mark.parametrize("model_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_templater_subject_first(model_name):
    template = "a [X] d e [Y] g."
    subject = "b  c "
    answer = "f"

    templater = Templater()
    text, spans = templater.replace_placeholders(template=template, subject=subject, answer=answer, capitalize=False)

    assert len(spans) == 2

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens, indices = templater.tokenize_with_span_dict(
        tokenizer=tokenizer, text=text, spans=spans, include_special_tokens=(model_name != "distilbert-base-cased")
    )

    assert len(indices) == 3

    for k, v in spans.items():
        log.info("%s: %s", k, ",".join((text[s:e] for s, e in v)))

    log.info(dict(enumerate(tokenizer.convert_ids_to_tokens(tokens["input_ids"]))))
    log.info(indices)

    assert text == "a b  c  d e f g."

    if model_name == "gpt2":
        assert indices["subject"] == [1, 2, 3, 4]  # extra space is encoded as token
        assert indices["answer"] == [7]
        assert indices["template"] == [0, 5, 6, 8, 9]
    elif model_name == "distilbert-base-cased":
        assert indices["subject"] == [1, 2]
        assert indices["answer"] == [5]
        assert indices["template"] == [0, 3, 4, 6, 7]
    elif model_name == "facebook/opt-125m":
        assert indices["subject"] == [2, 3, 4, 5]
        assert indices["answer"] == [8]
        assert indices["template"] == [1, 6, 7, 9, 10]
    else:
        msg = f"No check defined for {model_name}."
        raise RuntimeError(msg)


@pytest.mark.parametrize("model_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_evaluator_fill_template(model_name):
    evaluator = Evaluator.from_model(model_name)
    template = "[X] is the capital of [Y]. BERT. You could also: The capital of [Y] is [X]."
    expected_text = "Paris is the capital of France. BERT. You could also: The capital of France is paris."

    subject = "paris"
    answer = "France"

    assert evaluator.fill_template(template, subject=subject, answer=answer) == expected_text
    assert evaluator.fill_template(template, subject=subject, answer=answer, return_prefix=True) == (
        expected_text[:24],
        expected_text[24:],
    )
    assert evaluator.fill_template(
        template, subject=subject, answer=answer, return_prefix=True, return_suffix=True
    ) == (expected_text[:24], expected_text[24:75], expected_text[75:])

    _, indices = evaluator.fill_template(template, subject=subject, answer=answer, return_token_indices=True)
    assert len(indices) == 3

    if model_name == "gpt2":
        assert indices["subject"] == [0, 19, 20]  # Upper case is single token, lower case is split in two
        assert indices["answer"] == [5, 17]
    elif model_name == "distilbert-base-cased":
        assert indices["subject"] == [0, 20, 21]
        assert indices["answer"] == [5, 18]
    elif model_name == "facebook/opt-125m":
        assert indices["subject"] == [1, 20, 21]
        assert indices["answer"] == [6, 18]
    else:
        msg = f"No check defined for {model_name}."
        raise RuntimeError(msg)


@pytest.mark.parametrize("model_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_evaluator_fill_template_without_answer(model_name):
    evaluator = Evaluator.from_model(model_name)

    text, indices = evaluator.fill_template(
        template="a b c d.",
        answer="f",
        return_token_indices=True,
    )

    tokens = evaluator.tokenizer.tokenize(text)

    log.info(dict(enumerate(tokens)))

    assert text == "A b c d."

    if model_name in ("distilbert-base-cased", "gpt2"):
        assert indices["template"] == [0, 1, 2, 3, 4]
    elif model_name == "facebook/opt-125m":
        assert indices["template"] == [1, 2, 3, 4, 5]
    else:
        msg = f"No check defined for {model_name}."
        raise RuntimeError(msg)

    assert indices["subject"] == []
    assert indices["answer"] == []


@pytest.mark.parametrize("model_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_evaluator_fill_template_without_subject(model_name):
    evaluator = Evaluator.from_model(model_name)

    text, indices = evaluator.fill_template(
        template="a b c d e [Y]  g.",
        answer="f",
        return_token_indices=True,
    )

    tokens = evaluator.tokenizer.tokenize(text)

    log.info(dict(enumerate(tokens)))

    assert text == "A b c d e f  g."

    if model_name == "gpt2":
        assert indices["template"] == [0, 1, 2, 3, 4, 6, 7, 8]
        assert indices["answer"] == [5]
    elif model_name == "distilbert-base-cased":
        assert indices["template"] == [0, 1, 2, 3, 4, 6, 7]
        assert indices["answer"] == [5]
    elif model_name == "facebook/opt-125m":
        assert indices["template"] == [1, 2, 3, 4, 5, 7, 8, 9]
        assert indices["answer"] == [6]
    else:
        msg = f"No check defined for {model_name}."
        raise RuntimeError(msg)

    assert indices["subject"] == []


@pytest.mark.parametrize("model_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_evaluator_fill_template_subject_first(model_name):
    evaluator = Evaluator.from_model(model_name)

    template = "a [X] d e [Y] g."
    subject = "b  c "
    answer = "f"

    text, indices = evaluator.fill_template(
        template=template,
        subject=subject,
        answer=answer,
        return_token_indices=True,
        capitalize=False,
    )

    tokens = evaluator.tokenizer.tokenize(text)

    log.info(dict(enumerate(tokens)))

    assert text == "a b  c  d e f g."

    if model_name == "gpt2":
        assert indices["subject"] == [1, 2, 3, 4]  # extra space is encoded as token
        assert indices["answer"] == [7]
        assert indices["template"] == [0, 5, 6, 8, 9]
    elif model_name == "distilbert-base-cased":
        assert indices["subject"] == [1, 2]
        assert indices["answer"] == [5]
        assert indices["template"] == [0, 3, 4, 6, 7]
    elif model_name == "facebook/opt-125m":
        assert indices["subject"] == [2, 3, 4, 5]
        assert indices["answer"] == [8]
        assert indices["template"] == [1, 6, 7, 9, 10]
    else:
        msg = f"No check defined for {model_name}."
        raise RuntimeError(msg)


@pytest.mark.parametrize("model_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_evaluator_fill_template_object_first(model_name):
    evaluator = Evaluator.from_model(model_name)

    text, indices = evaluator.fill_template(
        template="a [Y] D E [X] G.",
        subject="F",
        answer="B C",
        return_token_indices=True,
    )

    assert text == "A B C D E F G."

    if model_name in ("gpt2", "distilbert-base-cased"):
        assert indices["answer"] == [1, 2]
        assert indices["subject"] == [5]
        assert indices["template"] == [0, 3, 4, 6, 7]
    elif model_name == "facebook/opt-125m":
        assert indices["answer"] == [2, 3]
        assert indices["subject"] == [6]
        assert indices["template"] == [1, 4, 5, 7, 8]
    else:
        msg = f"No check defined for {model_name}."
        raise RuntimeError(msg)


@pytest.mark.parametrize("model_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_evaluator_fill_template_no_prefix(model_name):
    evaluator = Evaluator.from_model(model_name)

    text, indices = evaluator.fill_template(
        template="[Y] C D E [X] G.",
        subject="F",
        answer="a b",
        return_token_indices=True,
    )

    assert text == "A b C D E F G."

    if model_name in ("gpt2", "distilbert-base-cased"):
        assert indices["answer"] == [0, 1]
        assert indices["subject"] == [5]
        assert indices["template"] == [2, 3, 4, 6, 7]
    elif model_name == "facebook/opt-125m":
        assert indices["answer"] == [1, 2]
        assert indices["subject"] == [6]
        assert indices["template"] == [3, 4, 5, 7, 8]
    else:
        msg = f"No check defined for {model_name}."
        raise RuntimeError(msg)


@pytest.mark.parametrize("model_name", ["distilbert-base-cased", "gpt2", "facebook/opt-125m"])
def test_evaluator_indices(model_name):
    evaluator = Evaluator.from_model(model_name)

    template = "a [Y] D E [X] G."
    subject = "F"
    answer = "B C"

    # List[Tuple[List[Tuple[str, float]], Dict[str, List[int]]]]
    instance_return_value = evaluator.evaluate_instance(
        template=template, answers=[answer], subject=subject, reduction=None
    )[0]

    assert not isinstance(instance_return_value, (float, int))

    token_scores, indices = instance_return_value

    if model_name == "facebook/opt-125m":
        # opt add an additional token
        assert len(token_scores) == sum(map(len, indices.values())) + 1
    elif model_name in ("distilbert-base-cased", "gpt2"):
        assert len(token_scores) == sum(map(len, indices.values()))
    else:
        msg = f"No check defined for {model_name}."
        raise RuntimeError(msg)

    tokens, _ = zip(*token_scores)

    log.debug(tokens)
    log.debug(indices)

    joined_tokens = {k: " ".join(tokens[i] for i in v) for k, v in indices.items()}

    assert joined_tokens["subject"] == subject
    assert joined_tokens["answer"] == answer
    assert joined_tokens["template"] == "A D E G ."
