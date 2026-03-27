import os
import json
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.model import SimpleCNN   # ✅ import your model

def evaluate(model_path, data_dir, output_dir):
    # Load trained model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Prepare test dataset
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

    # Run evaluation
    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()

    metrics = {
        "accuracy": acc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm
    }

    # Save metrics JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f)

    print("Evaluation complete. Accuracy:", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/opt/ml/processing/model/model.pth")
    parser.add_argument("--data_dir", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/model")
    args = parser.parse_args()

    evaluate(args.model_path, args.data_dir, args.output_dir)
