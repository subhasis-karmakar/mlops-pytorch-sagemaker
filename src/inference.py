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
        return torch.tensor(instances, dtype=torch.float32)

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        return outputs


def output_fn(prediction, accept):
    if accept == "application/json":
        predicted_classes = torch.argmax(prediction, dim=1).tolist()
        return json.dumps({"predictions": predicted_classes}), accept

    raise ValueError(f"Unsupported accept type: {accept}")