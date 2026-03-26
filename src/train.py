import argparse, os, torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
from model import SimpleCNN

def train(batch_size, epochs, lr, model_dir, data_dir, checkpoint_dir):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, checkpoint_path)

    torch.save(model.state_dict(), f"{model_dir}/model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--checkpoint_dir", type=str, default="/opt/ml/checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args.batch_size, args.epochs, args.lr, args.model_dir, args.data_dir, args.checkpoint_dir)