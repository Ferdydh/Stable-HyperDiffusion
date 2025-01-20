from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from einops import rearrange


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 12,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_model = d_model
        self.dropout = dropout

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Project to query, key, values
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        # Rearrange to multiple heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_head)

        # Efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0,
        )

        # Re-assemble all head outputs side by side
        y = rearrange(y, "b h n d -> b n (h d)")

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 12,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.attn = SelfAttention(d_model, n_head, dropout, bias)
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model=d_model, dropout=dropout, bias=bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_layer: int = 12,
        n_head: int = 12,
        d_model: int = 768,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.transformer = nn.ModuleList(
            [Block(d_model, n_head, dropout, bias) for _ in range(n_layer)]
        )

    def forward(self, x, mask=None):
        for block in self.transformer:
            x = block(x, mask)
        return x


####################################################################################################

"""
Taken from SANE repository
"""


class PositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs."""

    def __init__(self, max_positions: List[int], embedding_dim: int):
        """
        Args:
            max_positions: Maximum number of positions to embed for each dimension
            embedding_dim: Dimension of the embeddings
        """
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim

        if len(max_positions) == 2:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)
            self.pe3 = None
        elif len(max_positions) == 3:
            self.pe1 = nn.Embedding(max_positions[0], embedding_dim // 2)
            self.pe2 = nn.Embedding(max_positions[1], embedding_dim // 2)
            self.pe3 = nn.Embedding(max_positions[2], embedding_dim // 2)

    def forward(self, inputs: Tensor, positions: Tensor) -> Tensor:
        """
        Apply positional embeddings to input.

        Args:
            inputs: Input tensor [batch_size, seq_len, emb_dim]
            positions: Position tensors [batch_size, seq_len, n_dims]

        Returns:
            Output tensor with positional embeddings added
        """
        assert inputs.ndim == 3, f"Expected 3 dimensions, got {inputs.ndim}"
        assert (
            positions.shape[2] == len(self.max_positions)
        ), f"Position dimensions {positions.shape[2]} doesn't match max_positions {len(self.max_positions)}"
        assert positions.shape[0] == inputs.shape[0], "Batch size mismatch"
        assert positions.shape[1] == inputs.shape[1], "Sequence length mismatch"

        pos_emb1 = self.pe1(positions[:, :, 0])
        pos_emb2 = self.pe2(positions[:, :, 1])

        if self.pe3 is not None:
            pos_emb3 = self.pe3(positions[:, :, 2])
            pos_emb = [pos_emb1 + pos_emb2, pos_emb3]
        else:
            pos_emb = [pos_emb1, pos_emb2]

        pos_emb = torch.cat(pos_emb, dim=2)
        return inputs + pos_emb


class Encoder(nn.Module):
    """Encodes input tokens into latent distributions."""

    def __init__(
        self,
        max_positions: List[int],
        num_layers: int,
        d_model: int,
        dropout: float,
        num_heads: int,
        input_dim: int,
        latent_dim: int,
        layer_norm: bool
    ):
        super(Encoder, self).__init__()

        self.tokenizer = nn.Linear(input_dim, d_model)
        self.position_embeddings = PositionEmbs(
            max_positions=max_positions, embedding_dim=d_model
        )
        self.transformer = Transformer(
            n_layer=num_layers,
            n_head=num_heads,
            d_model=d_model,
            dropout=dropout,
            bias=False,
        )

        if layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            self.layer_norm = None

        # Separate projectors for mean and log variance
        self.mean_projector = nn.Linear(d_model, latent_dim)
        self.logvar_projector = nn.Linear(d_model, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Perform reparameterization trick to sample from N(mu, var) distribution.

        Args:
            mu: Mean of the latent Gaussian [batch_size, seq_len, latent_dim]
            logvar: Log variance of the latent Gaussian [batch_size, seq_len, latent_dim]

        Returns:
            z: Sampled latent vectors [batch_size, seq_len, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(
        self,
        input_tokens: Tensor,
        positions: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode input tokens into latent distributions.

        Args:
            input_tokens: Input token sequence [batch_size, seq_len, input_dim]
            positions: Position encodings [batch_size, seq_len, n_pos_dims]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            z: Sampled latent vectors [batch_size, seq_len, latent_dim]
            mu: Mean of the latent Gaussian [batch_size, seq_len, latent_dim]
            logvar: Log variance of the latent Gaussian [batch_size, seq_len, latent_dim]
        """
        embedded = self.tokenizer(input_tokens)
        if self.layer_norm:
            embedded = self.layer_norm(embedded)
        positioned = self.position_embeddings(embedded, positions)
        transformed = self.transformer(positioned, mask=attention_mask)

        # Get distributional parameters
        mu = self.mean_projector(transformed)
        logvar = self.logvar_projector(transformed)

        # Sample latent vectors using reparameterization trick
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class Decoder(nn.Module):
    """Decodes latent representations back to token space."""

    def __init__(
        self,
        max_positions: List[int],
        num_layers: int,
        d_model: int,
        dropout: float,
        num_heads: int,
        input_dim: int,
        latent_dim: int,
    ):
        super(Decoder, self).__init__()

        self.latent_projector = nn.Linear(latent_dim, d_model)
        self.position_embeddings = PositionEmbs(
            max_positions=max_positions, embedding_dim=d_model
        )
        self.transformer = Transformer(
            n_layer=num_layers,
            n_head=num_heads,
            d_model=d_model,
            dropout=dropout,
            bias=False,
        )
        self.detokenizer = nn.Linear(d_model, input_dim)

    def forward(
        self, latent: Tensor, positions: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Decode latent representations back to token space.

        Args:
            latent: Sampled latent vectors [batch_size, seq_len, latent_dim]
            positions: Position encodings [batch_size, seq_len, n_pos_dims]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            reconstructed: Reconstructed token sequence [batch_size, seq_len, input_dim]
        """
        decoded = self.latent_projector(latent)
        positioned = self.position_embeddings(decoded, positions)
        transformed = self.transformer(positioned, mask=attention_mask)
        reconstructed = self.detokenizer(transformed)
        return reconstructed
