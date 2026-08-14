from __future__ import annotations

from typing import Dict

import torch
from torch import nn

try:
    from transformers import AutoConfig, AutoModel
except ImportError:  # pragma: no cover
    AutoConfig = None
    AutoModel = None


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def penet_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def calculate_normalized_cosine_similarity(
    gallery_input: torch.Tensor,
    query: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    if query.dim() == 1:
        query = query.unsqueeze(0)
    if gallery_input.dim() == 1:
        gallery_input = gallery_input.unsqueeze(0)
    similarity = gallery_input @ query.transpose(0, 1)
    norm_1 = gallery_input.norm(dim=1, p=2).clamp(min=eps)
    norm_2 = query.norm(dim=1, p=2).clamp(min=eps)
    norm = norm_1.unsqueeze(1) * norm_2.unsqueeze(1).transpose(0, 1)
    return similarity / norm


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str, cache_dir: str | None = None, local_files_only: bool = False) -> None:
        super().__init__()
        if AutoModel is None or AutoConfig is None:
            raise ImportError("transformers is required for the hazard reasoning dual encoder.")
        try:
            self.backbone = AutoModel.from_pretrained(
                pretrained_model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except OSError:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.backbone = AutoModel.from_config(config)

    @property
    def hidden_size(self) -> int:
        return int(self.backbone.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]


class DualEncoderModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        projection_hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.1,
        share_backbone: bool = False,
        init_temperature: float = 0.07,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.scene_encoder = TextEncoder(
            pretrained_model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.rule_encoder = self.scene_encoder if share_backbone else TextEncoder(
            pretrained_model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        hidden_size = self.scene_encoder.hidden_size
        self.scene_projection = ProjectionHead(hidden_size, projection_hidden_dim, projection_dim, dropout)
        self.rule_projection = ProjectionHead(hidden_size, projection_hidden_dim, projection_dim, dropout)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temperature)))
        self.scene_projection_norm = nn.LayerNorm(projection_dim)
        self.rule_projection_norm = nn.LayerNorm(projection_dim)

    def encode_scene(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.scene_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        projected = self.scene_projection(features)
        projected = self.scene_projection_norm(projected)
        return penet_l2_normalize(projected)

    def encode_rule(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.rule_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        projected = self.rule_projection(features)
        projected = self.rule_projection_norm(projected)
        return penet_l2_normalize(projected)

    def compute_similarity(self, scene_embeddings: torch.Tensor, rule_embeddings: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        cosine_similarity = calculate_normalized_cosine_similarity(scene_embeddings, rule_embeddings)
        return scale * cosine_similarity

    @staticmethod
    def aggregate_major_logits(
        prototype_logits: torch.Tensor,
        prototype_major_label_ids: torch.Tensor,
        num_major_labels: int,
    ) -> torch.Tensor:
        batch_size = prototype_logits.size(0)
        major_logits = prototype_logits.new_full((batch_size, num_major_labels), float("-inf"))
        for label_id in range(num_major_labels):
            mask = prototype_major_label_ids == label_id
            if mask.any():
                major_logits[:, label_id] = prototype_logits[:, mask].max(dim=1).values
        return major_logits


def move_token_batch_to_device(token_batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in token_batch.items()}


@torch.no_grad()
def predict_major_labels(
    model: DualEncoderModel,
    scene_tokens: Dict[str, torch.Tensor],
    prototype_tokens: Dict[str, torch.Tensor],
    prototype_major_label_ids: torch.Tensor,
    num_major_labels: int,
) -> Dict[str, torch.Tensor]:
    scene_embeddings = model.encode_scene(scene_tokens)
    rule_embeddings = model.encode_rule(prototype_tokens)
    prototype_logits = model.compute_similarity(scene_embeddings, rule_embeddings)
    major_logits = model.aggregate_major_logits(
        prototype_logits=prototype_logits,
        prototype_major_label_ids=prototype_major_label_ids,
        num_major_labels=num_major_labels,
    )
    return {
        "prototype_logits": prototype_logits,
        "major_logits": major_logits,
        "pred_major_label_ids": major_logits.argmax(dim=1),
    }
