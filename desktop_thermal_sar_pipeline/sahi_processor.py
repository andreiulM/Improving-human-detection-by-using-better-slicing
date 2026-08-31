from __future__ import annotations
import logging
from typing import List, Optional
import cv2
import numpy as np

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, POSTPROCESS_NAME_TO_CLASS
from sahi.prediction import PredictionResult, ObjectPrediction
from sahi.slicing import slice_image

from desktop_thermal_sar_pipeline.config import PipelineConfig
from desktop_thermal_sar_pipeline.detector import Detection

logger = logging.getLogger(__name__)

class SAHIProcessor:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._detection_model: Optional[AutoDetectionModel] = None

    def initialize(self, model_path: str) -> None:
        logger.info("Initializing SAHI AutoDetectionModel from: %s", model_path)
        self._detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=self.config.model_confidence,
            device=self.config.model_device,
        )
        logger.info(
            "SAHI model ready — slice=%dx%d, postprocess=%s(%s@%.2f)",
            self.config.sahi_slice_width,
            self.config.sahi_slice_height,
            self.config.sahi_postprocess_type,
            self.config.sahi_postprocess_match_metric,
            self.config.sahi_postprocess_match_threshold,
        )
        if self.config.warmup_frames > 0:
            self._warmup()

    def _warmup(self) -> None:
        logger.info("Warming up GPU with %d dummy frames...", self.config.warmup_frames)
        dummy = np.random.randint(
            0, 255, 
            (self.config.frame_height, self.config.frame_width, 3), 
            dtype=np.uint8
        )
        for _ in range(self.config.warmup_frames):
            get_prediction(dummy, self._detection_model, verbose=0)
        logger.info("GPU warmup complete.")

    def detect_standard(self, frame: np.ndarray) -> List[Detection]:
        self._check_initialized()
        result = get_prediction(
            image=frame,
            detection_model=self._detection_model,
            verbose=0,
        )
        return self._convert(result)

    def detect_sahi(self, frame: np.ndarray) -> List[Detection]:
        self._check_initialized()
        img_h, img_w = frame.shape[:2]
        scale_w = img_w / self.config.frame_width
        scale_h = img_h / self.config.frame_height
        slice_h = int(self.config.sahi_slice_height * scale_h)
        slice_w = int(self.config.sahi_slice_width * scale_w)
        
        slice_image_result = slice_image(
            image=frame,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=self.config.sahi_overlap_ratio_h,
            overlap_width_ratio=self.config.sahi_overlap_ratio_w,
        )
        
        yolo_model = self._detection_model.model
        images = slice_image_result.images
        results = []
        for img in images:
            results.extend(
                yolo_model(
                    img, 
                    device=self.config.model_device, 
                    conf=self.config.model_confidence, 
                    verbose=False
                )
            )
        
        object_prediction_list = []
        for i, result in enumerate(results):
            shift_amount = slice_image_result.starting_pixels[i]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                x1 += shift_amount[0]
                x2 += shift_amount[0]
                y1 += shift_amount[1]
                y2 += shift_amount[1]
                
                object_prediction_list.append(
                    ObjectPrediction(
                        bbox=[x1, y1, x2, y2],
                        category_id=int(cls),
                        score=float(score),
                        category_name=self._detection_model.category_mapping.get(str(int(cls)), str(int(cls))),
                        shift_amount=[0,0]
                    )
                )

        if self.config.sahi_perform_standard_pred:
            standard_result = get_prediction(
                image=frame,
                detection_model=self._detection_model,
                shift_amount=[0, 0],
                full_shape=[frame.shape[0], frame.shape[1]],
                verbose=0
            )
            object_prediction_list.extend(standard_result.object_prediction_list)

        postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[self.config.sahi_postprocess_type]
        postprocess = postprocess_constructor(
            match_threshold=self.config.sahi_postprocess_match_threshold,
            match_metric=self.config.sahi_postprocess_match_metric,
            class_agnostic=self.config.sahi_postprocess_class_agnostic,
        )
        
        if len(object_prediction_list) > 1:
            object_prediction_list = postprocess(object_prediction_list)
            
        mock_result = PredictionResult(
            image=frame,
            object_prediction_list=object_prediction_list
        )
            
        return self._convert(mock_result)

    def _check_initialized(self) -> None:
        if self._detection_model is None:
            raise RuntimeError("SAHIProcessor not initialized. Call initialize() first.")

    def _convert(self, result: PredictionResult) -> List[Detection]:
        detections: List[Detection] = []
        for pred in result.object_prediction_list:
            class_id = int(pred.category.id)
            if class_id not in self.config.target_class_ids:
                continue
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(pred.score.value),
                    class_id=class_id,
                    class_name=str(pred.category.name),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
