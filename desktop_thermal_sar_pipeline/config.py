from __future__ import annotations
import logging
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PipelineConfig:
    frame_width: int = 640
    frame_height: int = 512
    input_channels: int = 1
    model_path: str = "None"
    model_confidence: float = 0.001
    model_iou_threshold: float = 0.5
    model_device: str = ""
    model_imgsz: int = 640
    half_precision: bool = True
    force_pytorch: bool = False
    sahi_enabled: bool = True
    slicer_backend: str = "microslicer"
    sahi_slice_width: int = 320
    sahi_slice_height: int = 256
    sahi_overlap_ratio_w: float = 0.25
    sahi_overlap_ratio_h: float = 0.25
    sahi_batch_size: int = 4
    sahi_perform_standard_pred: bool = True
    sahi_postprocess_type: str = "NMM"
    sahi_postprocess_match_metric: str = "IOS"
    sahi_postprocess_match_threshold: float = 0.5
    sahi_postprocess_class_agnostic: bool = True
    max_target_size_px: int = 30
    warmup_frames: int = 5
    metrics_window_size: int = 30
    draw_detections: bool = True
    bbox_color: tuple = (0, 255, 0)
    bbox_thickness: int = 2
    show_metrics_overlay: bool = True
    display_window_name: str = "Thermal SAR Pipeline"
    target_class_ids: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if not self.model_device:
            resolved = "cuda:0" if torch.cuda.is_available() else "cpu"
            object.__setattr__(self, "model_device", resolved)
            logger.info("Auto-detected device: %s", resolved)

    @classmethod
    def for_desktop(cls, gpu_id: int = 0) -> PipelineConfig:
        return cls(
            model_device=f"cuda:{gpu_id}",
            half_precision=True,
            warmup_frames=5,
        )

    def summary(self) -> str:
        lines = [
            "=" * 60,
            " THERMAL SAR PIPELINE",
            "=" * 60,
            f"  Device             : {self.model_device}",
            f"  Model              : {self.model_path}",
            f"  FP16               : {self.half_precision}",
            f"  Input Resolution   : {self.frame_width}×{self.frame_height}",
            f"  Confidence Thresh  : {self.model_confidence}",
            f"  Slicer Backend     : {self.slicer_backend if self.sahi_enabled else 'Disabled'}",
            f"  Slice Size         : {self.sahi_slice_width}×{self.sahi_slice_height}",
            f"  Target Classes     : {self.target_class_ids}",
            "=" * 60,
        ]
        return "\n".join(lines)
