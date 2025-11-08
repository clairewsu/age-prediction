import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.2, use_layernorm=False):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.LayerNorm(hidden_sizes[0]) if use_layernorm else nn.BatchNorm1d(hidden_sizes[0])
        self.drop1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.LayerNorm(hidden_sizes[1]) if use_layernorm else nn.BatchNorm1d(hidden_sizes[1])
        self.drop2 = nn.Dropout(dropout_prob)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.LayerNorm(hidden_sizes[2]) if use_layernorm else nn.BatchNorm1d(hidden_sizes[2])
        self.drop3 = nn.Dropout(dropout_prob)

        self.out = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.bn1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = F.gelu(x)
        x = self.bn2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = F.gelu(x)
        x = self.bn3(x)
        x = self.drop3(x)

        x = self.out(x)
        return x
