import torch
import torch.nn as nn


class mlp(nn.Module):
    def __init__(self, embed_dim=2):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, embed_dim)


    def forward(self, x):
        out = self.fc1(x.view(x.shape[0], -1))
        out = self.relu(out)
        out = self.fc2(out)
        return out