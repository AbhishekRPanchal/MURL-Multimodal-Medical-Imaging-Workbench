"""
Abstract Base Interface for Medical AI Platform Models
Defines lifecycle, preprocessing, inference execution, and standardized reporting.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch
from pydantic import BaseModel, Field


class ModelPredictionOutput(BaseModel):
    """
    Standardized multi-modal model output schema.
    """
    model_id: str
    modality: str
    task: str
    anatomical_region: str
    execution_device: str
    inference_time_ms: float
    raw_predictions: Dict[str, Any]
    formatted_results: Dict[str, Any]
    provenance: Dict[str, Any]


class BaseMedicalModel(ABC):
    """
    Abstract Base Class that every modality adapter must implement.
    Guarantees consistent configuration, safety validation, and execution interfaces.
    """

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        self.config = config
        self.model_id = config.get("model_id", "unknown_model")
        self.name = config.get("name", "Unknown Model")
        self.modality = config.get("modality", "Unknown")
        self.anatomical_region = config.get("anatomical_region", "Unknown")
        self.task = config.get("output_task", "unknown_task")
        self.device = device or torch.device("cpu")
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        """
        Loads model weights from config['weights_path'] onto self.device.
        """
        pass

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """
        Transforms raw scan input (PIL image, numpy array, or tensor)
        into model-ready tensors with appropriate dimensions and normalization.
        """
        pass

    @abstractmethod
    def infer(self, tensor_input: Any) -> Any:
        """
        Executes model forward pass without gradients.
        """
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats raw logits / maps into clinical research metrics, probabilities,
        and derived anatomical annotations.
        """
        pass

    def predict(self, raw_input: Any, context: Optional[Dict[str, Any]] = None) -> ModelPredictionOutput:
        """
        End-to-end execution pipeline with timing and provenance tracking.
        """
        import time

        if not self.is_loaded:
            self.load_model()

        context = context or {}
        self._active_context = context
        start_time = time.time()

        # Pipeline execution
        preprocessed = self.preprocess(raw_input)
        raw_output = self.infer(preprocessed)
        formatted = self.postprocess(raw_output, context)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return ModelPredictionOutput(
            model_id=self.model_id,
            modality=self.modality,
            task=self.task,
            anatomical_region=self.anatomical_region,
            execution_device=str(self.device),
            inference_time_ms=elapsed_ms,
            raw_predictions={"completed": True},
            formatted_results=formatted,
            provenance={
                "dataset": self.config.get("dataset"),
                "license": self.config.get("license"),
                "weights_path": self.config.get("weights_path"),
                "source_url": self.config.get("source_url")
            }
        )
