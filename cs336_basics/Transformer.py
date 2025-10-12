
import token
from turtle import forward
from typing import List, Dict, Tuple
from collections import Counter, deque
from torch import nn
import torch
import einops

'''
@Author: hychen11
@Date:   2025-10-09 11:48:00
@Description: 
'''


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Xavier initialization
        # Linear weights:  N(0,std^2) truncated at [-3std,3std]
        std = (2 / (in_features + out_features))**0.5
        self.weight = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
        # ... means the prefix dimension, such as (batch, seq, )


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        Given a sequence of token IDs, the Transformer language model uses a token embedding layer to produce a sequence of vectors. 
        Each embedding layer takes in a tensor of integers of shape (batch_size, sequence_length) and produces a sequence of vectors of shape (batch_size, sequence_length, d_model)
        """
        return self.weight[token_ids]


# llama layer norm : RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(
            d_model, device=device, dtype=dtype))
        # 这里的nn.init.ones_() 就是把公式里的g初始化成1
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_square = x*x
        mean_square = (einops.reduce(
            x_square, "batch_size seq_len d_model -> batch_size seq_len 1", "mean")+self.eps)**0.5
        result = x/mean_square * self.weight
        # Return the result in the original dtype
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = (d_ff + 63) // 64 * 64
        self.device = device
        self.dtype = dtype
        # Linear(d_in, d_out)
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.
        """
        swish = self.w1.forward(x)
        silu = swish * torch.sigmoid(swish)
        gate = self.w3.forward(x)
        out = silu * gate
        return self.w2.forward(out)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # dimension is d_k/2
        freqs = 1/(theta**(torch.arange(0, d_k, 2, device=device)/d_k))
        position_ids = torch.arange(
            0, max_seq_len, 1, dtype=torch.float32, device=device)  # dimension is seq_len

        freqs = torch.outer(position_ids, freqs)  # {seq_len,d_k/2}

        sin_freqs = torch.sin(freqs)
        cos_freqs = torch.cos(freqs)

        self.register_buffer(
            "cos_freqs", cos_freqs.to(device), persistent=False)
        self.register_buffer(
            "sin_freqs", sin_freqs.to(device), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        x along the sequence dimension.
        """
        seq_len, d_k = x.shape[-2:]
        x_pairs = einops.rearrange(
            x, "... seq_len (pairs two) -> ... seq_len pairs two", two=2)

        x_real = x_pairs[..., 0]
        x_imag = x_pairs[..., 1]

        # token position (..., seq_len)
        cos_freqs = self.cos_freqs[token_positions]  # (..., seq_len, pairs)
        sin_freqs = self.sin_freqs[token_positions]
        # insert a dimension at position 1 (for num_heads)
        # (..., num_heads, seq_len, pairs)
        cos_freqs = cos_freqs.unsqueeze(-3)  # (..., 1, seq_len, pairs)
        sin_freqs = sin_freqs.unsqueeze(-3)  # (..., 1, seq_len, pairs)

        rotated_real = x_real * cos_freqs - x_imag * sin_freqs
        rotated_imag = x_real * sin_freqs + x_imag * cos_freqs

        x_pairs = torch.stack([rotated_real, rotated_imag], dim=-1)

        rotated_x = einops.rearrange(
            x_pairs, "... seq_len pairs two -> ... seq_len (pairs two)", two=2)

        return rotated_x


def softmax(v: torch.Tensor, dim: int):
    # dim=i means compute at last dim
    maxv = torch.max(v, dim=dim, keepdim=True).values
    v_exp = torch.exp(v-maxv)
    return v_exp/torch.sum(v_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    keys and queries of shape (batch_size, ..., seq_len, d_k) 
    values of shape (batch_size, ..., seq_len, d_v)
    Optional mask: True should collectively sum to 1, and the attention probabilities of positions with a mask value of False should be zero.

    output with the shape (batch_size, ..., d_v)
    """
    batch_size = Q.shape[0]
    seq_len, d_k = Q.shape[-2:]
    d_v = V.shape[-1]
    mid = einops.einsum(
        Q, K, " ... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k")
    if mask is not None:
        mid = mid.masked_fill(~mask, float("-inf"))
    mid = mid/(d_k**0.5)
    res = softmax(mid, dim=-1)
    attention = einops.einsum(
        res, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return attention


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None, token_positions: torch.Tensor | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions
        self.d_k = self.d_v = d_model // num_heads

        if theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

        self.q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = einops.rearrange(
            Q, "batch_size seq_len (num_heads d_k)->batch_size num_heads seq_len d_k", num_heads=self.num_heads)
        K = einops.rearrange(
            K, "batch_size seq_len (num_heads d_k)->batch_size num_heads seq_len d_k", num_heads=self.num_heads)
        V = einops.rearrange(
            V, "batch_size seq_len (num_heads d_k)->batch_size num_heads seq_len d_k", num_heads=self.num_heads)

        if self.theta is not None:
            assert self.token_positions is not None, "token_positions must be provided if theta is not None"
            Q = self.rope.forward(Q, self.token_positions)
            K = self.rope.forward(K, self.token_positions)

        batch_size, num_heads, seq_len = Q.shape[0], Q.shape[1], Q.shape[2]

        mask = ~torch.triu(torch.ones(batch_size, num_heads, seq_len,
                           seq_len, device=Q.device, dtype=torch.bool), diagonal=1)

        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = einops.rearrange(
            attention, "batch_size num_heads seq_len d_k->batch_size seq_len (num_heads d_k)")
        return self.o(attention)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        token_positions = torch.zeros((1, max_seq_len), dtype=torch.long)
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, theta, max_seq_len, token_positions, device=device, dtype=dtype)

        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            token_positions = torch.arange(
                seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Update attention's token positions
        self.attn.token_positions = token_positions

        y = x + self.attn(self.ln1(x))
        z = y + self.ffn(self.ln2(y))
        return z