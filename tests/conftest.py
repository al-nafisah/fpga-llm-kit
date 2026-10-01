from pathlib import Path

import pytest

from fpga_llm_kit import hub
from fpga_llm_kit.model import Model

MODELS = Path(__file__).parent / "models"


@pytest.fixture
def load_model():
    """A model from its snapshot in tests/models, so the tests never touch the network."""
    def load(name: str) -> Model:
        config, tensors = hub.load(str(MODELS / f"{name}.json"))
        return Model.from_checkpoint(name, config, tensors)
    return load


@pytest.fixture
def snapshot():
    """The raw (config, tensors) of a snapshot, to break on purpose."""
    def load(name: str):
        return hub.load(str(MODELS / f"{name}.json"))
    return load
