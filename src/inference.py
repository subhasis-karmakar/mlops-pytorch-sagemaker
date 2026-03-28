import json
import os
import sys

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from model import SimpleCNN

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def model_fn(model_dir):
    model = SimpleCNN()
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model


def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)

        if "instances" not in data:
            raise ValueError("Missing 'instances' key in request payload")

        return torch.tensor(data["instances"], dtype=torch.float32)

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    with torch.no_grad():
        return model(input_data)


def output_fn(prediction, accept):
    if accept == "application/json":
        predicted_ids = torch.argmax(prediction, dim=1).tolist()
        predicted_labels = [CLASS_NAMES[i] for i in predicted_ids]

        return json.dumps(
            {
                "predictions": predicted_ids,
                "labels": predicted_labels,
            }
        ), accept

    raise ValueError(f"Unsupported accept type: {accept}")