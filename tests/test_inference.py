import torch
import os
import pytest
from src.inference import model_fn, predict_fn
from src.model import SimpleCNN

@pytest.fixture
def dummy_model_dir(tmp_path):
    # Create a temporary model checkpoint
    model = SimpleCNN()
    dummy_weights = model.state_dict()
    model_path = tmp_path / "model.pth"
    torch.save(dummy_weights, model_path)
    return tmp_path

def test_model_fn_loads_model(dummy_model_dir):
    model = model_fn(str(dummy_model_dir))
    assert isinstance(model, SimpleCNN)
    assert not model.training  # should be in eval mode

def test_predict_fn_runs_forward(dummy_model_dir):
    model = model_fn(str(dummy_model_dir))
    # Create dummy input: batch of 2 RGB images, 32x32
    dummy_input = torch.randn(2, 3, 32, 32)
    output = predict_fn(dummy_input, model)
    assert output.shape[0] == 2  # batch size preserved