import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
import torchvision
from ultralytics import YOLO, RTDETR

from desktop_thermal_sar_pipeline.config import PipelineConfig
from desktop_thermal_sar_pipeline.detector import Detection

logger = logging.getLogger(__name__)

class MicroSlicer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.model = None
        self._cached_slices: Optional[List[Tuple[int, int, int, int]]] = None
        self._last_h: int = 0
        self._last_w: int = 0
        self._class_names = None
        self._is_rtdetr: bool = False
        self._initialized = False

    def initialize(self, model_path: str) -> None:
        logger.info("Initializing MicroSlicer (Single-Threaded Sequential) from: %s", model_path)
        
        if "rtdetr" in model_path.lower():
            self.model = RTDETR(model_path)
            self._is_rtdetr = True
        else:
            self.model = YOLO(model_path, task="detect")
            self._is_rtdetr = False

        self._initialized = True
            
        logger.info(
            "MicroSlicer ready (Sequential) — slice=%dx%d, iou_thresh=%.2f",
            self.config.sahi_slice_width,
            self.config.sahi_slice_height,
            self.config.model_iou_threshold
        )
        
        if self.config.warmup_frames > 0:
            self._warmup()

    def _run_inference(self, img: np.ndarray) -> list:
        results = self.model(
            img, 
            device=self.config.model_device, 
            conf=self.config.model_confidence, 
            verbose=False
        )
        if self._class_names is None and len(results) > 0:
            self._class_names = results[0].names
        return results

    def _warmup(self) -> None:
        logger.info("Warming up GPU with %d dummy frames...", self.config.warmup_frames)
        dummy = np.random.randint(
            0, 255, 
            (self.config.frame_height, self.config.frame_width, 3), 
            dtype=np.uint8
        )
        for _ in range(self.config.warmup_frames):
            self._run_inference(dummy)
        logger.info("GPU warmup complete.")

    def _get_slice_bboxes(self, image_h: int, image_w: int) -> List[Tuple[int, int, int, int]]:
        if self._cached_slices is not None and image_h == self._last_h and image_w == self._last_w:
            return self._cached_slices
            
        scale_w = image_w / self.config.frame_width
        scale_h = image_h / self.config.frame_height
        
        slice_h = int(self.config.sahi_slice_height * scale_h)
        slice_w = int(self.config.sahi_slice_width * scale_w)
        
        step_h = int(slice_h * (1 - self.config.sahi_overlap_ratio_h))
        step_w = int(slice_w * (1 - self.config.sahi_overlap_ratio_w))
        
        bboxes = []
        y = 0
        while y < image_h:
            x = 0
            while x < image_w:
                x_max = min(x + slice_w, image_w)
                y_max = min(y + slice_h, image_h)
                x_min = max(0, x_max - slice_w)
                y_min = max(0, y_max - slice_h)
                bboxes.append((x_min, y_min, x_max, y_max))
                if x + slice_w >= image_w:
                    break
                x += step_w
            if y + slice_h >= image_h:
                break
            y += step_h
            
        self._cached_slices = bboxes
        self._last_h = image_h
        self._last_w = image_w
        return bboxes

    def detect_standard(self, frame: np.ndarray) -> List[Detection]:
        self._check_initialized()
        results = self._run_inference(frame)[0]
        return self._parse_results([results], [(0, 0, frame.shape[1], frame.shape[0])])

    def detect_sahi(self, frame: np.ndarray) -> List[Detection]:
        self._check_initialized()
        h, w = frame.shape[:2]
        slice_bboxes = self._get_slice_bboxes(h, w)
        
        all_detections = []
        
        for (x1, y1, x2, y2) in slice_bboxes:
            crop = frame[y1:y2, x1:x2]
            results = self._run_inference(crop)
            batch_detections = self._parse_results(results, [(x1, y1, x2, y2)])
            all_detections.extend(batch_detections)
            
        return self._apply_nms(all_detections)

    def _parse_results(self, results_list: list, bboxes: List[Tuple[int, int, int, int]]) -> List[Detection]:
        detections = []
        for results, (x1, y1, x2, y2) in zip(results_list, bboxes):
            if results.boxes is not None and len(results.boxes) > 0:
                data = results.boxes.data.cpu().numpy()
                for row in data:
                    conf = float(row[4])
                    cls_id = int(row[5])
                    
                    if self._is_rtdetr or cls_id in self.config.target_class_ids:
                        mapped_id = 0 if self._is_rtdetr else cls_id
                        global_bbox = (
                            float(row[0] + x1), 
                            float(row[1] + y1), 
                            float(row[2] + x1), 
                            float(row[3] + y1)
                        )
                        
                        detections.append(Detection(
                            bbox=global_bbox,
                            confidence=conf,
                            class_id=mapped_id,
                            class_name=self._class_names.get(mapped_id, "person") if self._class_names else "person"
                        ))
        return detections

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
            
        boxes_tensor = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
        scores_tensor = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
        
        keep_indices = torchvision.ops.nms(
            boxes_tensor, 
            scores_tensor, 
            iou_threshold=self.config.model_iou_threshold
        )
        
        return [detections[i] for i in keep_indices.tolist()]

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MicroSlicer not initialized. Call initialize() first.")
