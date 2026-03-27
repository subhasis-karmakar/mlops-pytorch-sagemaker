import os
import json
import argparse
import tarfile
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.model import SimpleCNN


def extract_model_artifact(model_tar_path: str, extract_dir: str) -> str:
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    # Find model.pth
    for root, _, files in os.walk(extract_dir):
        if "model.pth" in files:
            return os.path.join(root, "model.pth")

    raise FileNotFoundError("model.pth not found in artifact")


def evaluate(model_artifact_path, data_dir, output_dir):
    extracted_dir = "/opt/ml/processing/model_extracted"
    model_path = extract_model_artifact(model_artifact_path, extracted_dir)

    # Load model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    preds, labels = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.numpy())
            labels.extend(targets.numpy())

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()

    metrics = {
        "evaluation": {
            "accuracy": float(acc),
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
            "confusion_matrix": cm
        }
    }

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f)

    print("Evaluation complete:", acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--data_dir", default="/opt/ml/processing/test")
    parser.add_argument("--output_dir", default="/opt/ml/processing/evaluation")

    args = parser.parse_args()

    evaluate(args.model_path, args.data_dir, args.output_dir)