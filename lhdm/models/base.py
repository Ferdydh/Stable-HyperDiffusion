from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from typeguard import typechecked


class Encoder(nn.Module):
    @typechecked
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, **kwargs):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.dropout = nn.Dropout(0.1)

    @typechecked
    def forward(
        self, x: Float[Tensor, "batch input_dim"]
    ) -> Float[Tensor, "batch z_dim"]:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        z = self.fc2(x)
        return z


class Decoder(nn.Module):
    @typechecked
    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int, **kwargs):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    @typechecked
    def forward(
        self, z: Float[Tensor, "batch z_dim"]
    ) -> Float[Tensor, "batch output_dim"]:
        z = F.relu(self.fc1(z))
        z = self.dropout(z)
        x_reconstructed = self.fc2(z)
        return x_reconstructed
