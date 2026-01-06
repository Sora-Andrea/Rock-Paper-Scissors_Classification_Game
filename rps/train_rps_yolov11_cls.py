import os
from ultralytics import YOLO

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset
DATASET_DIR = os.path.join(BASE_DIR, "rps_split_dataset")

IMG_SIZE = 320
EPOCHS = 25
BATCH = 32

PROJECT = "rps_yolo11_cls"
NAME = "yolo11x_rps"

def main():
    model = YOLO("yolo11x-cls.yaml").load("yolo11x-cls.pt")

    # Train
    results = model.train(
        task="classify",
        data=DATASET_DIR,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        augment=True,
        hsv_h=0.01,
        hsv_s=0.20,
        hsv_v=0.20,
        degrees=5.0,
        translate=0.02,
        scale=0.05,
        fliplr=0.25,
    )

    # validate after training
    metrics = model.val(task="classify", data=DATASET_DIR, imgsz=IMG_SIZE)
    print(metrics)

if __name__ == "__main__":
    main()
