
from typing import List, Dict, Tuple
from collections import Counter, deque
from torch import nn
import torch
import einops
# from einops import rearrange, einsum

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
