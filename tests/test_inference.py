import json
import torch
import pytest

from src.inference import model_fn, input_fn, predict_fn, output_fn
from src.model import SimpleCNN


@pytest.fixture
def dummy_model_dir(tmp_path):
    model = SimpleCNN()
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)
    return tmp_path


def test_model_fn_loads_model(dummy_model_dir):
    model = model_fn(str(dummy_model_dir))
    assert isinstance(model, SimpleCNN)
    assert not model.training


def test_input_fn_parses_json():
    payload = {
        "instances": [
            [[[0.1] * 32 for _ in range(32)] for _ in range(3)],
            [[[0.2] * 32 for _ in range(32)] for _ in range(3)],
        ]
    }

    result = input_fn(json.dumps(payload), "application/json")

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.shape == (2, 3, 32, 32)


def test_input_fn_raises_for_unsupported_content_type():
    with pytest.raises(ValueError, match="Unsupported content type"):
        input_fn("{}", "text/plain")


def test_predict_fn_returns_class_ids(dummy_model_dir):
    model = model_fn(str(dummy_model_dir))
    dummy_input = torch.randn(2, 3, 32, 32)

    output = predict_fn(dummy_input, model)

    assert isinstance(output, list)
    assert len(output) == 2
    assert all(isinstance(x, int) for x in output)


def test_output_fn_returns_json_response():
    prediction = [1, 3]
    body, content_type = output_fn(prediction, "application/json")

    assert content_type == "application/json"

    parsed = json.loads(body)
    assert "predictions" in parsed
    assert parsed["predictions"] == [1, 3]


def test_output_fn_raises_for_unsupported_accept():
    with pytest.raises(ValueError, match="Unsupported accept type"):
        output_fn([1], "text/plain")