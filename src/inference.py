import json
import torch

from src.model import SimpleCNN


def model_fn(model_dir):
    model = SimpleCNN()
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model


def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        instances = data["instances"]
        tensor = torch.tensor(instances, dtype=torch.float32)
        return tensor

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
        return predicted.tolist()


def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps({"predictions": prediction}), accept

    raise ValueError(f"Unsupported accept type: {accept}")