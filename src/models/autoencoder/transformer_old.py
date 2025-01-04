"""
Code mostly taken / inspired by Andrey Kaparthy's NanoGpt\
https://github.com/karpathy/nanoGPT/blob/master/model.py

His References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


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
        causal: bool = False,
        block_size: int = 128,
    ):
        super().__init__()
        assert d_model % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = d_model
        self.dropout = dropout
        self.causal = causal
        self.block_size = block_size
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # generates a lower triangular matrices. Used to maks out 'future' information from attention matrix
            if self.causal:
                self.register_buffer(
                    "bias",
                    torch.tril(torch.ones(block_size, block_size)).view(
                        1, 1, block_size, block_size
                    ),
                )
        else:
            # all three methods should be enables anyways, but let's make it explicit here
            torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            )

    def forward(self, x, mask=None):
        (
            B,  # B: batch_size
            T,  # T: token_number
            C,  # C: token_embedding_dim
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # move head forward to be the batch dim
        k = k.view(
            B, T, self.n_head, C // self.n_head
        ).transpose(
            1, 2
        )  # (B, nh, T, hs) # (nH = self.n_head), hs = C//nh (token_embedding_dim per head)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.causal,  # we currently don't want / need autoregressive / causal attention
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(
                    self.bias[:, :, :T, :T] == 0, float("-inf")
                )  # causal attention: masking future tokens, upper right triangular matrix
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
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
        causal: bool = False,
        block_size: int = 128,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.attn = SelfAttention(d_model, n_head, dropout, bias, causal, block_size)
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
        causal: bool = False,
        block_size: int = 128,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.transformer = nn.ModuleList(
            [
                Block(d_model, n_head, dropout, bias, causal, block_size)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, mask=None):
        for block in self.transformer:
            x = block(x, mask)
        return x


####################################################################################################

"""
Taken from SANE repository
"""


class ProjectionHead(nn.Module):
    """Projects token embeddings to a lower-dimensional space."""

    def __init__(self, d_model: int, n_tokens: int, output_dim: int):
        """
        Args:
            d_model: Dimension of the input embeddings
            n_tokens: Number of tokens in the sequence
            output_dim: Dimension of the output projection
        """
        super(ProjectionHead, self).__init__()

        input_dim = d_model * n_tokens
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Projects token embeddings to output dimension.

        Args:
            embeddings: Token embeddings [batch_size, n_tokens, d_model]

        Returns:
            projected: Projected embeddings [batch_size, output_dim]
        """
        batch_embeddings = embeddings.view(embeddings.shape[0], -1)
        projected = self.projection(batch_embeddings)
        return projected


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
    """Encodes input tokens into latent representations."""

    def __init__(
        self,
        max_positions: List[int],
        num_layers: int,
        d_model: int,
        dropout: float,
        window_size: int,
        num_heads: int,
        input_dim: int,
        latent_dim: int,
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
            causal=False,
            block_size=window_size,
        )
        self.latent_projector = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        input_tokens: Tensor,
        positions: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode input tokens into latent representations.

        Args:
            input_tokens: Input token sequence [batch_size, seq_len, input_dim]
            positions: Position encodings [batch_size, seq_len, n_pos_dims]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            latent: Encoded latent representations [batch_size, seq_len, latent_dim]
        """
        embedded = self.tokenizer(input_tokens)
        positioned = self.position_embeddings(embedded, positions)
        transformed = self.transformer(positioned, mask=attention_mask)
        latent = self.latent_projector(transformed)
        return latent


class Decoder(nn.Module):
    """Decodes latent representations back to token space."""

    def __init__(
        self,
        max_positions: List[int],
        num_layers: int,
        d_model: int,
        dropout: float,
        window_size: int,
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
            causal=False,
            block_size=window_size,
        )
        self.detokenizer = nn.Linear(d_model, input_dim)

    def forward(
        self, latent: Tensor, positions: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Decode latent representations back to token space.

        Args:
            latent: Encoded latent representations [batch_size, seq_len, latent_dim]
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
