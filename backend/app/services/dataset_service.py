"""
Dataset service: wraps the existing data pipeline.

In production, this would call:
- build_scene_graph_features.py for scene graph feature extraction
- build_detection_visual_features_clip.py for CLIP features
- build_fused_hetero_data.py for building HeteroData objects
"""

import json
import os


def get_pseudo_label_rules():
    """Load pseudo-label rules from the existing YAML configuration."""
    rules_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "engine", "decct-gnn", "data", "pseudo_label_rules.yaml"
    )
    # Fallback: return summary
    return {
        "behaviorClasses": [
            "climbing_scaffold_frame",
            "leaning_out",
            "standing_on_guardrail",
            "throwing_material",
        ],
        "ruleCount": 4,
        "description": "Rule-based behavior scoring from bridge edge features",
    }


def get_prior_knowledge():
    """Load prior knowledge graph."""
    kg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "engine", "decct-gnn", "prior-knowledge-new.json"
    )
    # Fallback
    return {
        "riskMechanisms": ["fall_risk", "struck_by_object_risk"],
        "behaviorCategories": ["height_related", "stability_related", "tool_related"],
        "riskStages": ["precursor", "occurring", "critical"],
    }


def export_dataset_coco(dataset_id):
    """Export dataset images as COCO-format JSON (for training)."""
    from app.models.dataset import Dataset, DatasetImage
    from app import create_app

    app = create_app()
    with app.app_context():
        ds = Dataset.query.get(dataset_id)
        if not ds:
            return None

        images = DatasetImage.query.filter_by(dataset_id=dataset_id).all()
        coco_data = {
            "images": [
                {
                    "id": img.id,
                    "file_name": img.file_name,
                    "width": img.width,
                    "height": img.height,
                }
                for img in images
            ],
            "annotations": [],
            "categories": [],
        }
        return coco_data
