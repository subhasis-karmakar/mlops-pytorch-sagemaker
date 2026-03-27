import torch
import pytest
from src.model import SimpleCNN

def test_simplecnn_forward_shape():
    model = SimpleCNN()
    model.eval()

    # Dummy batch: 4 RGB images, 32x32
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)

    # Output should be [batch_size, num_classes]
    assert output.shape == (4, 10)

def test_simplecnn_grad_flow():
    model = SimpleCNN()
    dummy_input = torch.randn(2, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (2,))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Forward + backward pass
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    loss.backward()
    optimizer.step()

    # Ensure loss is a scalar and gradients exist
    assert loss.item() > 0
    for param in model.parameters():
        assert param.grad is not None