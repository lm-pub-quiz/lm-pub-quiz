from collections import namedtuple

import pytest
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

TEST_MODELS = {
    "distilbert": ("distilbert-base-cased", "MLM"),
    "distilgpt": ("distilgpt2", "CLM"),
}


ModelData = namedtuple("ModelData", ("model", "tokenizer", "name_or_path", "model_type"))


@pytest.fixture(scope="session")
def model_cache() -> dict[str, ModelData]:
    """Prepare the models used during testing.

    Since the scope is "session", the models are loaded only once.
    The models should only be used for evaluation. Refrain from editing these,
    as other test-cases rely on them.
    """

    cache: dict[str, ModelData] = {}

    for key, (name_or_path, model_type) in TEST_MODELS.items():
        if model_type == "MLM":
            model = AutoModelForMaskedLM.from_pretrained(name_or_path)
        elif model_type == "CLM":
            model = AutoModelForCausalLM.from_pretrained(name_or_path)
        else:
            msg = f"Model type {model_type} not implemented."
            raise ValueError(msg)

        cache[key] = ModelData(
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(name_or_path),
            name_or_path=name_or_path,
            model_type=model_type,
        )

    return cache


@pytest.fixture(scope="session")
def distilbert(model_cache):
    return model_cache["distilbert"].model, model_cache["distilbert"].tokenizer
