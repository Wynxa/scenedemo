from __future__ import annotations

import torch
from torch import nn


def make_fc(dim_in: int, hidden_dim: int, use_gn: bool = False) -> nn.Module:
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, nn.GroupNorm(32 if hidden_dim % 32 == 0 else 1, hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc
