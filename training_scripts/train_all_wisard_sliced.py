import os
from ultralytics import YOLO, RTDETR

DATASET_YAML = "training_yaml_files/sliced_wisard.yaml"

def train_yolo26n():
    print("==================================================")
    print("1. Starting YOLO26n Training on Sliced WiSARD Dataset")
    print("==================================================")
    model = YOLO("yolo26n.pt")
    model.train(
        data=DATASET_YAML,
        epochs=150,
        patience=50,
        batch=64,
        nbs=64,
        cache=False,
        imgsz=640,
        multi_scale=False,
        deterministic=False,
        device='0',
        workers=4,
        name="yolo26n_wisard_sliced_640px",
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        cos_lr=True,
        box=10.0,
        cls=5.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,
        translate=0.15,
        scale=0.1,
        erasing=0.0,
        mosaic=0.0,
        close_mosaic=0,
        exist_ok=True,
        iou=0.45,
        seed=0,
    )

def train_rtdetr_l():
    print("==================================================")
    print("2. Starting RT-DETR-L Training on Sliced WiSARD Dataset")
    print("==================================================")
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=DATASET_YAML,
        epochs=200,
        patience=50,
        batch=8,
        nbs=64,
        cache=False,
        imgsz=640,
        multi_scale=False,
        deterministic=False,
        device='0',
        workers=4,
        name="rtdetr_l_wisard_sliced_640px",
        optimizer="AdamW",
        lr0=0.0001,
        weight_decay=0.0005,
        cos_lr=True,
        freeze=5,
        dropout=0.1,
        iou=0.45,
        box=10.0,
        cls=5.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,
        translate=0.15,
        scale=0.1,
        erasing=0.0,
        mosaic=0.0,
        close_mosaic=0,
        exist_ok=True,
        seed=0,
    )

def train_yolo26s():
    print("==================================================")
    print("3. Starting YOLO26s Training on Sliced WiSARD Dataset")
    print("==================================================")
    model = YOLO("yolo26s.pt")
    model.train(
        data=DATASET_YAML,
        epochs=150,
        patience=50,
        batch=32,
        nbs=64,
        cache=False,
        imgsz=640,
        multi_scale=False,
        deterministic=False,
        device='0',
        workers=4,
        name="yolo26s_wisard_sliced_640px",
        optimizer="AdamW",
        lr0=0.0001,
        weight_decay=0.0005,
        cos_lr=True,
        iou=0.45,
        seed=0,
        box=10.0,
        cls=5.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,
        translate=0.15,
        scale=0.1,
        erasing=0.0,
        mosaic=0.0,
        close_mosaic=0,
        exist_ok=True,
    )

if __name__ == "__main__":
    print("Starting sliced training sequence for WiSARD dataset (320px native crops)...")
    train_yolo26n()
    train_rtdetr_l()
    train_yolo26s()
    print("All sliced training runs completed.")
