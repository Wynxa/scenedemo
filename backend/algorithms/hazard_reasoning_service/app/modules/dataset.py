from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch


class PrototypeInventory:
    def __init__(self, inventory_path: str | Path):
        self.inventory_path = Path(inventory_path)
        with self.inventory_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.meta = payload["meta"]
        self.summary = payload["summary"]
        self.prototypes = list(payload["prototypes"])
        self.prototype_id_to_index = {
            proto["prototype_id"]: idx for idx, proto in enumerate(self.prototypes)
        }
        self.major_label_to_id = self._build_major_label_map(self.prototypes)
        self.prototype_texts = [proto["text"] for proto in self.prototypes]
        self.prototype_major_label_ids = torch.tensor(
            [int(proto["major_label_id"]) for proto in self.prototypes],
            dtype=torch.long,
        )

    @staticmethod
    def _build_major_label_map(prototypes: List[Dict]) -> Dict[str, int]:
        mapping = {}
        for proto in prototypes:
            mapping[proto["major_label"]] = int(proto["major_label_id"])
        return mapping
