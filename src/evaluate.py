import os
import json
import argparse
import tarfile

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1)
        self.fc = nn.Linear(32 * 30 * 30, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def extract_model_artifact(model_tar_path: str, extract_dir: str) -> str:
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    for root, _, files in os.walk(extract_dir):
        if "model.pth" in files:
            return os.path.join(root, "model.pth")

    raise FileNotFoundError("model.pth not found inside model.tar.gz")


def evaluate(model_artifact_path: str, data_dir: str, output_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extracted_dir = "/opt/ml/processing/model_extracted"
    model_path = extract_model_artifact(model_artifact_path, extracted_dir)

    model = SimpleCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    preds = []
    labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())

    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    confusion = confusion_matrix(labels, preds).tolist()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": confusion
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    print("FINAL METRICS:", json.dumps(metrics))
    print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    print(f"Saved evaluation report to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--data_dir", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    evaluate(
        model_artifact_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )