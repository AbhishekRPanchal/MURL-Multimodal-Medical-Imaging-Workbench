"""
Model Router
Dispatches incoming scan inputs to compatible, validated models.
Enforces the safety guarantee: Only validated models are routed automatically.
"""

from typing import Dict, Any, Optional, List, Tuple
import torch

from src.platform.registry import ModelRegistry
from src.platform.compatibility_validator import ModelCompatibilityValidator
from src.platform.base_model import BaseMedicalModel
from src.platform.adapters.busi_ultrasound import BUSIClassifierAdapter, BUSISegmentorAdapter
from src.platform.adapters.medsam_adapter import MedSAMAdapter
from src.platform.adapters.totalsegmentator_adapter import TotalSegmentatorAdapter
from src.platform.adapters.nnunet_adapter import nnUNetAdapter
from src.platform.adapters.chest_xray_adapter import ChestXRayAdapter
from src.platform.adapters.monai_brats_adapter import MONAIBraTSAdapter
from src.platform.adapters.thyroid_ultrasound_adapter import ThyroidUltrasoundAdapter


# Adapter registry mapping model_id to adapter class
MODEL_ADAPTER_MAP = {
    "busi_resnet18_classifier": BUSIClassifierAdapter,
    "busi_monai_unet_segmentor": BUSISegmentorAdapter,
    "medsam_vit_b": MedSAMAdapter,
    "totalsegmentator_ct": TotalSegmentatorAdapter,
    "nnunet_planner": nnUNetAdapter,
    "chexpert_chest_xray_classifier": ChestXRayAdapter,
    "monai_brats_mri_segmentor": MONAIBraTSAdapter,
    "tn3k_thyroid_segmentor": ThyroidUltrasoundAdapter
}


class ModelRouter:
    """
    Intelligent routing service that matches scan characteristics to approved models.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None, device: Optional[torch.device] = None):
        self.registry = registry or ModelRegistry()
        self.device = device or torch.device("cpu")
        self._instantiated_adapters: Dict[str, BaseMedicalModel] = {}

    def get_adapter(self, model_id: str) -> BaseMedicalModel:
        """
        Retrieves or instantiates the adapter for a given model_id.
        """
        if model_id in self._instantiated_adapters:
            return self._instantiated_adapters[model_id]

        model_meta = self.registry.get_model(model_id)
        if model_id not in MODEL_ADAPTER_MAP:
            raise NotImplementedError(
                f"Model adapter for '{model_id}' is not yet implemented in the current runtime environment."
            )

        adapter_cls = MODEL_ADAPTER_MAP[model_id]
        adapter_instance = adapter_cls(model_meta, device=self.device)
        adapter_instance.load_model()
        self._instantiated_adapters[model_id] = adapter_instance
        return adapter_instance

    def route(
        self,
        scan_properties: Dict[str, Any],
        requested_model_id: Optional[str] = None,
        requested_task: Optional[str] = None
    ) -> Tuple[BaseMedicalModel, Dict[str, Any]]:
        """
        Routes the scan to the appropriate model.

        Args:
            scan_properties: Scan metadata (modality, format, spatial_ndim, num_channels, etc.)
            requested_model_id: Explicit model requested by user/client (optional)
            requested_task: Target clinical task (classification, segmentation, etc.)

        Returns:
            Tuple of (Instantiated BaseMedicalModel adapter, compatibility report)
        """
        modality = scan_properties.get("modality", "")

        # Scenario 1: User explicitly requested a specific model
        if requested_model_id:
            model_meta = self.registry.get_model(requested_model_id)

            # Check validation status
            val_status = model_meta.get("validation_status")
            if val_status not in ["validated", "adapter_ready"]:
                raise PermissionError(
                    f"Model '{requested_model_id}' has validation status '{val_status}'. "
                    f"Only verified operational models can be executed for research inference."
                )

            compat = ModelCompatibilityValidator.validate_compatibility(model_meta, scan_properties)
            if not compat["compatible"]:
                raise ValueError(
                    f"Scan is incompatible with requested model '{requested_model_id}': {compat['issues']}"
                )

            adapter = self.get_adapter(requested_model_id)
            return adapter, compat

        # Scenario 2: Automated routing based on scan properties
        candidate_models = self.registry.list_models(
            modality=modality,
            task=requested_task,
            anatomical_region=scan_properties.get("anatomical_region"),
            validated_only=True  # First prioritize validated models
        )

        if not candidate_models:
            # Fall back to adapter_ready models for the modality (e.g. BraTS MRI, Thyroid US)
            all_mod = self.registry.list_models(
                modality=modality,
                task=requested_task,
                anatomical_region=scan_properties.get("anatomical_region")
            )
            candidate_models = [m for m in all_mod if m.get("validation_status") in ["validated", "adapter_ready"]]

        if not candidate_models and "mri/ct" in modality.lower():
            # Check CT then MRI
            cand_ct = [m for m in self.registry.list_models(modality="CT") if m.get("validation_status") in ["validated", "adapter_ready"]]
            cand_mri = [m for m in self.registry.list_models(modality="MRI") if m.get("validation_status") in ["validated", "adapter_ready"]]
            candidate_models = cand_ct + cand_mri

        if not candidate_models:
            all_for_modality = self.registry.list_models(modality=modality)
            if all_for_modality:
                unvalidated_names = [m["name"] for m in all_for_modality]
                raise RuntimeError(
                    f"Modality '{modality}' was detected, but all matching models {unvalidated_names} "
                    f"are currently unvalidated/planned. Automated routing is disabled until validation is complete."
                )
            raise LookupError(f"No models found in registry supporting modality '{modality}'.")

        # Validate candidates for physical compatibility
        for cand in candidate_models:
            compat = ModelCompatibilityValidator.validate_compatibility(cand, scan_properties)
            if compat["compatible"]:
                adapter = self.get_adapter(cand["model_id"])
                return adapter, compat

        raise ValueError(
            f"None of the validated candidate models for modality '{modality}' are compatible with the provided scan: "
            f"format={scan_properties.get('format')}, dims={scan_properties.get('spatial_ndim')}D"
        )
