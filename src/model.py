import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1)
        self.fc = nn.Linear(32*30*30, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)