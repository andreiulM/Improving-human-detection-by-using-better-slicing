import os
import csv
import json
import logging
import queue
import threading
import time
from datetime import datetime
import argparse
import cv2
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from desktop_thermal_sar_pipeline.config import PipelineConfig
from desktop_thermal_sar_pipeline.pipeline import ThermalSARPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate_pipeline")

SIZE_BINS = [
    {"key": "h_lt8", "label": "h < 8 px", "min_h": 0.0, "max_h": 8.0},
    {"key": "h_8_16", "label": "8 <= h < 16 px", "min_h": 8.0, "max_h": 16.0},
    {"key": "h_16_32", "label": "16 <= h < 32 px", "min_h": 16.0, "max_h": 32.0},
    {"key": "h_ge32", "label": "h >= 32 px", "min_h": 32.0, "max_h": float("inf")},
]

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def get_size_bin_index(height_px: float) -> int:
    for idx, b in enumerate(SIZE_BINS):
        if b["min_h"] <= height_px < b["max_h"]:
            return idx
    return len(SIZE_BINS) - 1

class TargetSizeEvaluator:
    def __init__(self, iou_threshold: float = 0.50, conf: float = 0.25):
        self.iou_threshold = iou_threshold
        self.conf = conf
        self.bins = SIZE_BINS
        self.bin_stats = [{"total_gt": 0, "matched_tp": 0} for _ in self.bins]
        self.total_gt = 0
        self.total_tp = 0
        self.total_dt_at_conf = 0

    def evaluate_frame(self, gt_boxes_xyxy, gt_heights, pred_boxes_xyxy, pred_scores):
        num_gt = len(gt_boxes_xyxy)
        if num_gt == 0:
            for score in pred_scores:
                if score >= self.conf:
                    self.total_dt_at_conf += 1
            return

        gt_bin_indices = []
        for h in gt_heights:
            b_idx = get_size_bin_index(h)
            gt_bin_indices.append(b_idx)
            self.bin_stats[b_idx]["total_gt"] += 1
            self.total_gt += 1

        filtered_preds = []
        for box, score in zip(pred_boxes_xyxy, pred_scores):
            if score >= self.conf:
                filtered_preds.append((box, score))

        self.total_dt_at_conf += len(filtered_preds)

        if not filtered_preds:
            return

        filtered_preds.sort(key=lambda x: x[1], reverse=True)

        matched_gt_indices = set()
        for p_box, p_score in filtered_preds:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g_box in enumerate(gt_boxes_xyxy):
                if g_idx in matched_gt_indices:
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                b_idx = gt_bin_indices[best_gt_idx]
                self.bin_stats[b_idx]["matched_tp"] += 1
                self.total_tp += 1

    def get_summary(self) -> dict:
        results = {}
        for idx, b in enumerate(self.bins):
            gt_count = self.bin_stats[idx]["total_gt"]
            tp_count = self.bin_stats[idx]["matched_tp"]
            recall = (tp_count / gt_count) if gt_count > 0 else 0.0
            results[f"recall_{b['key']}"] = round(recall, 4)
            results[f"gt_{b['key']}"] = gt_count
            results[f"tp_{b['key']}"] = tp_count
            results[f"fn_{b['key']}"] = max(0, gt_count - tp_count)

        overall_recall = (self.total_tp / self.total_gt) if self.total_gt > 0 else 0.0
        overall_precision = (self.total_tp / self.total_dt_at_conf) if self.total_dt_at_conf > 0 else 0.0
        results["match_conf"] = self.conf
        results["match_iou_thresh"] = self.iou_threshold
        results["match_total_gt"] = self.total_gt
        results["match_total_tp"] = self.total_tp
        results["match_total_dt"] = self.total_dt_at_conf
        results["match_recall50"] = round(overall_recall, 4)
        results["match_precision50"] = round(overall_precision, 4)
        return results

def append_row_to_csv(csv_path: str, row: dict) -> None:
    parent_dir = os.path.dirname(os.path.abspath(csv_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    file_exists = os.path.isfile(csv_path)
    if not file_exists:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        dict_reader = csv.DictReader(f)
        existing_rows = list(dict_reader)
        existing_fieldnames = list(dict_reader.fieldnames) if dict_reader.fieldnames else []

    if existing_fieldnames and set(row.keys()) <= set(existing_fieldnames):
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=existing_fieldnames, extrasaction="ignore")
            writer.writerow(row)
    else:
        all_fieldnames = list(existing_fieldnames)
        for k in row.keys():
            if k not in all_fieldnames:
                all_fieldnames.append(k)

        existing_rows.append(row)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)

def parse_yolo_labels(lbl_path, img_w, img_h):
    bboxes = []
    if not os.path.exists(lbl_path):
        return bboxes

    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id != 0:
                    continue

                x_c = float(parts[1]) * img_w
                y_c = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h

                x_min = x_c - w / 2.0
                y_min = y_c - h / 2.0

                bboxes.append([x_min, y_min, w, h])
    return bboxes

class _PrefetchLoader:
    _SENTINEL = None

    def __init__(self, img_files, img_dir, lbl_dir, prefetch: int = 8):
        self._img_files = img_files
        self._img_dir = img_dir
        self._lbl_dir = lbl_dir
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        for img_name in self._img_files:
            img_path = os.path.join(self._img_dir, img_name)
            lbl_path = os.path.join(self._lbl_dir, os.path.splitext(img_name)[0] + ".txt")
            frame = cv2.imread(img_path)
            self._queue.put((img_name, frame, lbl_path))
        self._queue.put(self._SENTINEL)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                return
            yield item

def evaluate(
    dataset_path: str,
    model_path: str,
    mode: str = "adaptive",
    limit: int = 0,
    split: str = "test",
    slicer: str = "sahi",
    conf: float = 0.25,
    summary_csv: str = None,
):
    img_dir = os.path.join(dataset_path, "images", split)
    lbl_dir = os.path.join(dataset_path, "labels", split)

    if not os.path.exists(img_dir):
        if split == "test" and os.path.exists(os.path.join(dataset_path, "images", "val")):
            logger.info("Split 'test' not found. Falling back to 'val' split.")
            split = "val"
            img_dir = os.path.join(dataset_path, "images", split)
            lbl_dir = os.path.join(dataset_path, "labels", split)
        else:
            logger.error(f"Image directory not found: {img_dir}")
            return

    img_files = sorted(f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png")))
    if limit > 0:
        img_files = img_files[:limit]

    logger.info(f"Found {len(img_files)} images for evaluation.")

    config_kwargs = {
        "model_path": model_path,
        "half_precision": True,
        "target_class_ids": (0,),
        "slicer_backend": slicer,
        "model_confidence": conf,
    }

    if mode == "standard":
        config_kwargs["sahi_enabled"] = False
        logger.info("Evaluating STANDARD (No Slicing, No SR)")
    elif mode in ["force_slicing", "force_sahi"]:
        logger.info("Evaluating FORCED SLICING")

    config = PipelineConfig(**config_kwargs)

    coco_gt_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "Person"}],
    }

    coco_dt_list = []
    annotation_id = 1
    total_latency_ms = 0.0

    target_size_evaluator = TargetSizeEvaluator(iou_threshold=0.50, conf=conf)

    loader = _PrefetchLoader(img_files, img_dir, lbl_dir, prefetch=8)
    with ThermalSARPipeline(config) as pipeline:
        logger.info(f"Pipeline Initialized. Starting Inference (Prefetch loader active, Confidence: {conf})...")

        for img_id, (img_name, frame, lbl_path) in enumerate(loader, start=1):
            if frame is None:
                continue

            img_h, img_w = frame.shape[:2]

            coco_gt_dict["images"].append(
                {"id": img_id, "file_name": img_name, "width": img_w, "height": img_h}
            )

            gt_boxes = parse_yolo_labels(lbl_path, img_w, img_h)
            frame_gt_xyxy = []
            frame_gt_heights = []

            for box in gt_boxes:
                coco_gt_dict["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

                x_min, y_min, w, h = box
                frame_gt_xyxy.append([x_min, y_min, x_min + w, y_min + h])
                frame_gt_heights.append(h)

            t0 = time.perf_counter()
            result = pipeline.process_frame(frame)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if img_id > config.warmup_frames:
                total_latency_ms += latency_ms

            frame_pred_xyxy = []
            frame_pred_scores = []
            for det in result.detections:
                x1, y1, x2, y2 = det.bbox
                w, h = x2 - x1, y2 - y1
                if det.class_id == 0:
                    coco_dt_list.append(
                        {
                            "image_id": img_id,
                            "category_id": 0,
                            "bbox": [x1, y1, w, h],
                            "score": det.confidence,
                        }
                    )
                    frame_pred_xyxy.append([x1, y1, x2, y2])
                    frame_pred_scores.append(det.confidence)

            target_size_evaluator.evaluate_frame(
                frame_gt_xyxy, frame_gt_heights, frame_pred_xyxy, frame_pred_scores
            )

            if img_id % 100 == 0:
                logger.info(
                    f"Processed {img_id}/{len(img_files)} frames... Current pipeline mode: {result.mode}"
                )

    avg_latency = total_latency_ms / max(1, (len(img_files) - config.warmup_frames))
    size_summary = target_size_evaluator.get_summary()

    logger.info("Inference complete. Calculating COCO metrics...")

    gt_json_path = f"tmp_gt_{mode}.json"
    dt_json_path = f"tmp_dt_{mode}.json"

    with open(gt_json_path, "w") as f:
        json.dump(coco_gt_dict, f)
    with open(dt_json_path, "w") as f:
        json.dump(coco_dt_list, f)
    try:
        if not coco_dt_list:
            logger.warning("No detections were made across the tested frames. Evaluation metrics cannot be computed.")
            return {
                "mode": mode,
                "avg_latency": avg_latency,
                "map": 0.0,
                "map50": 0.0,
                "ap_small": 0.0,
                "recall": 0.0,
                "recall50": 0.0,
                "ar_small": 0.0,
                "size_summary": size_summary,
            }

        coco_gt = COCO(gt_json_path)
        coco_dt = coco_gt.loadRes(dt_json_path)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        try:
            recall50 = coco_eval.eval['recall'][0, 0, 0, 2]
            if recall50 == -1:
                recall50 = 0.0
        except Exception as e:
            raise RuntimeError("Failed to extract Recall@0.50 from COCO metrics.") from e

        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        map50_95 = coco_eval.stats[0] if len(coco_eval.stats) > 0 else 0.0
        map50 = coco_eval.stats[1] if len(coco_eval.stats) > 1 else 0.0
        ap_small = coco_eval.stats[3] if len(coco_eval.stats) > 3 else 0.0
        ar100 = coco_eval.stats[8] if len(coco_eval.stats) > 8 else 0.0
        ar_small = coco_eval.stats[9] if len(coco_eval.stats) > 9 else 0.0

        tp = size_summary["match_total_tp"]
        total_gt = size_summary["match_total_gt"]
        total_dt = size_summary["match_total_dt"]
        fn = max(0, total_gt - tp)
        fp = max(0, total_dt - tp)
        precision50 = size_summary["match_precision50"]
        coco_max_recall50 = recall50
        recall50 = size_summary["match_recall50"]
        f1_score = (2.0 * precision50 * recall50 / (precision50 + recall50)) if (precision50 + recall50) > 0 else 0.0

        logger.info("=" * 65)
        logger.info(f"  FINAL PIPELINE METRICS: {mode.upper()}")
        logger.info("=" * 65)
        logger.info(f"  Model               : {model_path}")
        logger.info(f"  Dataset Split       : {dataset_path} ({split}) - {len(img_files)} frames")
        logger.info(f"  Total Ground Truth  : {total_gt}")
        logger.info(f"  Total Predictions   : {total_dt} (at conf >= {conf})")
        logger.info("  " + "-" * 58)
        logger.info(f"  True Positives  (TP): {tp}")
        logger.info(f"  False Negatives (FN): {fn}")
        logger.info(f"  False Positives (FP): {fp}")
        logger.info("  " + "-" * 58)
        logger.info(f"  Precision@0.50      : {precision50 * 100:.2f}%")
        logger.info(f"  Recall@0.50         : {recall50 * 100:.2f}%")
        logger.info(f"  F1-Score@0.50       : {f1_score * 100:.2f}%")
        logger.info(f"  mAP@0.50            : {map50:.4f}")
        logger.info(f"  mAP@0.50:0.95       : {map50_95:.4f}")
        logger.info(f"  AP_small            : {ap_small:.4f}")
        logger.info(f"  AR_small            : {ar_small:.4f}")
        logger.info(f"  Average Latency     : {avg_latency:.2f} ms per frame ({fps:.1f} FPS)")
        logger.info("=" * 65)

        logger.info("=" * 65)
        logger.info(f"  TARGET-SIZE STRATIFIED RECALL@0.50 (Operational conf >= {conf}):")
        logger.info("=" * 65)
        logger.info("  %-18s %-12s %-14s %-14s" % ("Height Interval", "GT Targets", "Matched (TP)", "Recall@0.50"))
        logger.info("  " + "-" * 62)
        for b in SIZE_BINS:
            k = b["key"]
            gt_cnt = size_summary[f"gt_{k}"]
            tp_cnt = size_summary[f"tp_{k}"]
            rec = size_summary[f"recall_{k}"] * 100.0 if gt_cnt > 0 else 0.0
            rec_str = f"{rec:.2f}%" if gt_cnt > 0 else "N/A"
            logger.info("  %-18s %-12d %-14d %-14s" % (b["label"], gt_cnt, tp_cnt, rec_str))
        logger.info("  " + "-" * 62)
        logger.info("  %-18s %-12d %-14d %-14s" % (
            "Overall",
            size_summary["match_total_gt"],
            size_summary["match_total_tp"],
            f"{size_summary['match_recall50'] * 100.0:.2f}%",
        ))
        logger.info("=" * 65)

        if summary_csv:
            summary_csv_path = summary_csv
        elif abs(conf - 0.25) < 1e-4:
            summary_csv_path = "benchmark_evaluation_summary_homogenous_conf0.25.csv"
        elif abs(conf - 0.001) < 1e-4:
            summary_csv_path = "benchmark_evaluation_summary_homogenous_conf0.001.csv"
        else:
            summary_csv_path = f"benchmark_evaluation_summary_homogenous_conf{conf}.csv"

        model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path))) if "weights" in model_path else os.path.basename(model_path)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "model_path": model_path,
            "dataset": dataset_path,
            "split": split,
            "mode": mode,
            "slicer": "no" if mode == "standard" else slicer,
            "match_conf": conf,
            "total_frames": len(img_files),
            "total_gt": total_gt,
            "total_dt": total_dt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision50": round(precision50, 4),
            "recall50": round(recall50, 4),
            "f1_score": round(f1_score, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "ap_small": round(ap_small, 4),
            "ar_small": round(ar_small, 4),
            "recall_h_lt8": size_summary["recall_h_lt8"],
            "recall_h_8_16": size_summary["recall_h_8_16"],
            "recall_h_16_32": size_summary["recall_h_16_32"],
            "recall_h_ge32": size_summary["recall_h_ge32"],
            "gt_h_lt8": size_summary["gt_h_lt8"],
            "gt_h_8_16": size_summary["gt_h_8_16"],
            "gt_h_16_32": size_summary["gt_h_16_32"],
            "gt_h_ge32": size_summary["gt_h_ge32"],
            "tp_h_lt8": size_summary["tp_h_lt8"],
            "tp_h_8_16": size_summary["tp_h_8_16"],
            "tp_h_16_32": size_summary["tp_h_16_32"],
            "tp_h_ge32": size_summary["tp_h_ge32"],
            "avg_latency_ms": round(avg_latency, 2),
            "fps": round(fps, 1),
        }

        append_row_to_csv(summary_csv_path, row)
        logger.info("Benchmark summary appended to: %s", os.path.abspath(summary_csv_path))

        return {
            "mode": mode,
            "avg_latency": avg_latency,
            "map": map50_95,
            "map50": map50,
            "ap_small": ap_small,
            "recall": ar100,
            "recall50": recall50,
            "ar_small": ar_small,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision50": precision50,
            "f1_score": f1_score,
            "size_summary": size_summary,
            "recall_h_lt8": size_summary["recall_h_lt8"],
            "recall_h_8_16": size_summary["recall_h_8_16"],
            "recall_h_16_32": size_summary["recall_h_16_32"],
            "recall_h_ge32": size_summary["recall_h_ge32"],
            "coco_max_recall50": coco_max_recall50,
        }

    except Exception as e:
        import traceback
        logger.error(f"Failed to calculate COCO eval metrics: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Hard failure: Metric calculation failed for {mode}") from e

    finally:
        if os.path.exists(gt_json_path):
            os.remove(gt_json_path)
        if os.path.exists(dt_json_path):
            os.remove(dt_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Thermal SAR Detection Pipeline")
    parser.add_argument("--dataset", type=str, default="wisard_dataset")
    parser.add_argument("--model", type=str, default="runs/detect/rtdetr_l_thermal_sar/weights/best.pt")
    parser.add_argument("--mode", type=str, choices=["standard", "adaptive", "force_slicing", "force_sahi"], default="adaptive")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--slicer", type=str, choices=["sahi", "microslicer"], default="sahi")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--summary-csv", type=str, default=None)

    args = parser.parse_args()
    evaluate(
        dataset_path=args.dataset,
        model_path=args.model,
        mode=args.mode,
        limit=args.limit,
        split=args.split,
        slicer=args.slicer,
        conf=args.conf,
        summary_csv=args.summary_csv,
    )
