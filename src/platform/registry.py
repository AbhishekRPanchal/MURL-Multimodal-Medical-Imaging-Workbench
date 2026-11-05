"""
Model Registry
Manages registry.yaml metadata, schema validation, and access controls.
Enforces the strict rule: Only validated models can be selected automatically.
"""

import os
from typing import Dict, List, Any, Optional
import yaml

MANDATORY_REGISTRY_FIELDS = [
    "model_id",
    "name",
    "modality",
    "anatomical_region",
    "supported_image_formats",
    "required_input_dimensions",
    "num_input_channels",
    "required_mri_sequences",
    "output_task",
    "dataset",
    "weights_path",
    "source_url",
    "license",
    "training_provenance",
    "evaluation_metrics",
    "validation_status",
    "known_limitations"
]

VALID_VALIDATION_STATUSES = ["validated", "adapter_ready", "unvalidated", "planned"]
VALID_OUTPUT_TASKS = ["classification", "segmentation", "detection", "joint", "planning", "interactive_segmentation"]
VALID_MODALITIES = ["Ultrasound", "MRI", "CT", "X-ray", "Mammography", "PET", "Multi-Modal"]


class ModelRegistry:
    """
    Central repository for multi-modal model metadata.
    Guarantees strict schema validation and isolates planned/unvalidated models from automatic execution.
    """

    def __init__(self, registry_path: str = "model_registry/registry.yaml"):
        self.registry_path = registry_path
        self._models: Dict[str, Dict[str, Any]] = {}
        self.load_registry()

    def load_registry(self) -> None:
        """
        Loads and validates registry.yaml.
        """
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(f"Registry file not found at: {self.registry_path}")

        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "models" not in data:
            raise ValueError(f"Invalid registry format in {self.registry_path}: Missing 'models' root key.")

        models_list = data["models"]
        validated_dict = {}

        for item in models_list:
            self._validate_model_entry(item)
            model_id = item["model_id"]
            if model_id in validated_dict:
                raise ValueError(f"Duplicate model_id detected in registry: {model_id}")
            validated_dict[model_id] = item

        self._models = validated_dict

    def _validate_model_entry(self, entry: Dict[str, Any]) -> None:
        """
        Validates that a model entry contains all 17 mandatory fields and valid values.
        """
        for field in MANDATORY_REGISTRY_FIELDS:
            if field not in entry:
                raise ValueError(f"Model entry '{entry.get('model_id', 'UNKNOWN')}' is missing required field: '{field}'")

        if entry["validation_status"] not in VALID_VALIDATION_STATUSES:
            raise ValueError(
                f"Model '{entry['model_id']}' has invalid validation_status: '{entry['validation_status']}'. "
                f"Must be one of {VALID_VALIDATION_STATUSES}"
            )

        if entry["output_task"] not in VALID_OUTPUT_TASKS:
            raise ValueError(
                f"Model '{entry['model_id']}' has invalid output_task: '{entry['output_task']}'. "
                f"Must be one of {VALID_OUTPUT_TASKS}"
            )

        if not isinstance(entry["supported_image_formats"], list):
            raise ValueError(f"Model '{entry['model_id']}': supported_image_formats must be a list.")

        if not isinstance(entry["required_input_dimensions"], list):
            raise ValueError(f"Model '{entry['model_id']}': required_input_dimensions must be a list.")

    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Retrieves metadata for a specific model ID.
        """
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in registry. Available: {list(self._models.keys())}")
        return self._models[model_id]

    def list_models(
        self,
        modality: Optional[str] = None,
        task: Optional[str] = None,
        anatomical_region: Optional[str] = None,
        validated_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Lists models with optional filtering.
        """
        results = []
        for model in self._models.values():
            if validated_only and model["validation_status"] != "validated":
                continue
            if modality and model["modality"].lower() != modality.lower():
                continue
            if task and model["output_task"].lower() != task.lower():
                continue
            if anatomical_region and model["anatomical_region"].lower() != anatomical_region.lower() and model["anatomical_region"].lower() != "whole-body":
                continue
            results.append(model)
        return results

    def get_validated_models(self) -> List[Dict[str, Any]]:
        """
        Returns only models that are actively verified and eligible for automated routing.
        """
        return [m for m in self._models.values() if m["validation_status"] == "validated"]

    def is_automatically_routable(self, model_id: str) -> bool:
        """
        Checks whether the specified model is approved for automatic routing.
        """
        model = self.get_model(model_id)
        return model.get("validation_status") == "validated"
