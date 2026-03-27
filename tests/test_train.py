import torch
import os
import tempfile
import pytest
from src.model import SimpleCNN
from src.train import train

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=10):
        self.size = size
        self.data = torch.randn(size, 3, 32, 32)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as checkpoint_dir:
        yield model_dir, checkpoint_dir

def test_train_runs_and_saves_model(temp_dirs, monkeypatch):
    model_dir, checkpoint_dir = temp_dirs

    # Monkeypatch CIFAR10 dataset with dummy dataset
    monkeypatch.setattr("torchvision.datasets.CIFAR10", lambda *args, **kwargs: DummyDataset())

    # Run training for 1 epoch with small batch size
    train(batch_size=2, epochs=1, lr=0.001, model_dir=model_dir, data_dir=".", checkpoint_dir=checkpoint_dir)

    # Check that model.pth was saved
    model_path = os.path.join(model_dir, "model.pth")
    assert os.path.exists(model_path)

    # Load model weights to ensure file is valid
    model = SimpleCNN()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)