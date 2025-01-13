import os

import torch
from my_project.model import BaselineModel
import pytest

# VARIABLES

# test data

def test_model():
    model = BaselineModel()
    assert model is not None, "Model was not created successfully"
    assert isinstance(model, torch.nn.Module), "Model is not an instance of torch.nn.Module"

    dummy_input = torch.randn(1, 1, 28, 28)
    #with pytest.raises(ValueError, match="Expected input to be a 4D tensor"):
    output = model(dummy_input)
    assert output.shape == (1, 10), f"The output shpae of the model is incorrect. Expected (1, 10), got {output.shape}"
