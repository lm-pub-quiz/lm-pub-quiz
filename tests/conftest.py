import pytest
from transformers import AutoModelForMaskedLM, AutoTokenizer


@pytest.fixture(scope="session")
def distilbert():
    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    return model, tokenizer
