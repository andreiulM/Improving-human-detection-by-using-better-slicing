from __future__ import annotations
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from desktop_thermal_sar_pipeline.config import PipelineConfig
from desktop_thermal_sar_pipeline.detector import Detection
from desktop_thermal_sar_pipeline.sahi_processor import SAHIProcessor
from desktop_thermal_sar_pipeline.micro_slicer import MicroSlicer
from desktop_thermal_sar_pipeline.metrics import CUDATimer, FrameMetrics, MetricsTracker

logger = logging.getLogger(__name__)

MODE_STANDARD = "STANDARD"
MODE_SAHI = "SAHI"

@dataclass
class PipelineResult:
    detections: List[Detection]
    annotated: np.ndarray
    metrics: FrameMetrics
    mode: str
    raw_frame: np.ndarray

class ThermalSARPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._slicer = None
        self._metrics: Optional[MetricsTracker] = None
        self._frame_counter: int = 0
        self._is_initialized: bool = False

    def initialize(self) -> None:
        logger.info("\n%s", self.config.summary())
        if self.config.slicer_backend == "microslicer":
            self._slicer = MicroSlicer(self.config)
        else:
            self._slicer = SAHIProcessor(self.config)
        
        self._slicer.initialize(self.config.model_path)
        self._metrics = MetricsTracker(window_size=self.config.metrics_window_size)
        self._is_initialized = True
        logger.info("Pipeline initialization complete")

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize before process_frame.")
            
        slicer_label = "MICROSLICER" if self.config.slicer_backend == "microslicer" else "SAHI"

        if frame.ndim == 2:
            frame_3ch = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame_3ch = cv2.cvtColor(frame.squeeze(2), cv2.COLOR_GRAY2BGR)
        else:
            frame_3ch = frame

        self._frame_counter += 1
        timer = CUDATimer(self.config.model_device)
        
        with timer:
            if self.config.sahi_enabled:
                detections = self._slicer.detect_sahi(frame_3ch)
                mode = slicer_label
            else:
                detections = self._slicer.detect_standard(frame_3ch)
                mode = MODE_STANDARD
                
        latency_ms = timer.elapsed_ms
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        frame_metrics = FrameMetrics(
            frame_id=self._frame_counter,
            latency_ms=latency_ms,
            fps=fps,
            mode=mode,
            num_detections=len(detections)
        )
        self._metrics.record(frame_metrics)
        annotated = self._annotate_frame(frame_3ch, detections, frame_metrics)

        return PipelineResult(
            detections=detections,
            annotated=annotated,
            metrics=frame_metrics,
            mode=mode,
            raw_frame=frame
        )

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        metrics: FrameMetrics,
    ) -> np.ndarray:
        annotated = frame.copy()
        
        if not self.config.draw_detections:
            return annotated
            
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            color = self.config.bbox_color
            
            if det.is_small_target(self.config.max_target_size_px):
                color = (0, 255, 255)
                
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.config.bbox_thickness)
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
            
        if self.config.show_metrics_overlay:
            self._draw_hud(annotated, metrics)
            
        return annotated

    def _draw_hud(self, frame: np.ndarray, metrics: FrameMetrics) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        mode_color = (0, 255, 0)
        if "MICROSLICER" in metrics.mode or "SAHI" in metrics.mode:
            mode_color = (0, 200, 255)

        lines = [
            (f"Mode: {metrics.mode}", mode_color),
            (f"FPS:  {metrics.fps:.1f}", (255, 255, 255)),
            (f"Lat:  {metrics.latency_ms:.1f} ms", (255, 255, 255)),
            (f"Dets: {metrics.num_detections}", (255, 255, 255)),
        ]

        y_offset = 20
        for text, color in lines:
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            y_offset += 20

    @property
    def metrics_tracker(self) -> MetricsTracker:
        if self._metrics is None:
            raise RuntimeError("Pipeline not initialized.")
        return self._metrics

    @property
    def frame_count(self) -> int:
        return self._frame_counter

    def release(self) -> None:
        logger.info("Releasing pipeline resources...")
        if self._metrics is not None:
            logger.info("\n%s", self._metrics.report())
        cv2.destroyAllWindows()
        self._is_initialized = False
        logger.info("Pipeline resources released.")

    def __enter__(self) -> "ThermalSARPipeline":
        self.initialize()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
