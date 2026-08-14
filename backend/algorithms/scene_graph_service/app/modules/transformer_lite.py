from __future__ import annotations

import torch
from torch import nn


class TransformerEncoderHTCLLite(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_head: int,
        d_k: int,
        d_v: int,
        d_model: int,
        d_inner: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _ = d_k
        _ = d_v
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_inner,
            dropout=dropout,
            batch_first=True,
        )
        # Avoid PyTorch's prototype nested-tensor fast path.  It only emits a
        # warning on some Windows/GPU builds, but disabling it also keeps the
        # scene-graph runtime on the stable Transformer execution path.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

    def forward(self, input_feats: torch.Tensor, num_objs: list[int]) -> torch.Tensor:
        if not num_objs:
            return input_feats
        if max(num_objs) < 80:
            split_feats = input_feats.split(num_objs, dim=0)
            padded_feats = nn.utils.rnn.pad_sequence(split_feats, batch_first=True)
            bsz = len(num_objs)
            device = padded_feats.device
            pad_len = max(num_objs)
            num_objs_tensor = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
            non_pad_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).lt(num_objs_tensor)
            obj_pad_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_tensor)
            encoded = self.transformer_encoder(padded_feats, src_key_padding_mask=obj_pad_mask)
            return encoded[non_pad_mask]

        original_input_feats = input_feats
        output = torch.zeros_like(input_feats)
        id_list = list(range(input_feats.shape[0]))
        if min(num_objs) < 80:
            short_idx = [i for i, x in enumerate(num_objs) if x < 80]
            short_list: list[int] = []
            for i in short_idx:
                start = sum(num_objs[:i])
                short_list.extend(id_list[start:start + num_objs[i]])
            num_short = [num_objs[i] for i in short_idx]
            short_feats = input_feats[short_list].split(num_short, dim=0)
            short_feats = nn.utils.rnn.pad_sequence(short_feats, batch_first=True)
            bsz = len(num_short)
            device = short_feats.device
            pad_len = max(num_short)
            num_short_tensor = torch.LongTensor(num_short).to(device).unsqueeze(1).expand(-1, pad_len)
            non_pad_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).lt(num_short_tensor)
            obj_pad_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_short_tensor)
            short_output = self.transformer_encoder(short_feats, src_key_padding_mask=obj_pad_mask)
            output[short_list] = short_output[non_pad_mask]

        long_idx = [i for i, x in enumerate(num_objs) if x >= 80]
        for i in long_idx:
            start = sum(num_objs[:i])
            end = start + num_objs[i]
            encoded = self.transformer_encoder(original_input_feats[start:end].unsqueeze(0))
            output[start:end] = encoded.squeeze(0)
        return output
