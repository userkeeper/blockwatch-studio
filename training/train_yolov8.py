"""
Train YOLOv8n on the BlockWatch Studio dataset.

Run from the `training/` directory after labelling images with
LabelImg in YOLO format and placing them under `../dataset/`.

    cd training/
    python -m pip install ultralytics            # one-time
    python train_yolov8.py

Output:
    training/runs/detect/train/weights/best.pt   # PyTorch weights
    training/runs/detect/train/weights/best.onnx # ONNX export
"""
from pathlib import Path
import shutil

from ultralytics import YOLO


# YOLOv8 'n' (nano) — 6 MB, ~40 ms / 640x640 inference on CPU.
# Big enough for our ~6-class problem, small enough to bundle in
# the installer.
PRETRAINED = "yolov8n.pt"

# Training hyperparameters tuned for a small (~500 image) dataset.
EPOCHS = 80
BATCH_SIZE = 16
IMAGE_SIZE = 640
PATIENCE = 20            # early-stop if val loss plateaus
DEVICE = "cpu"            # change to "0" for first NVIDIA GPU


def main() -> None:
    model = YOLO(PRETRAINED)
    results = model.train(
        data="dataset.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        project="runs",
        name="train",
        exist_ok=False,
    )

    # Export the best weights to ONNX so the Rust runtime can load it.
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    export_model = YOLO(str(best_pt))
    onnx_path = export_model.export(format="onnx", opset=17, dynamic=False)

    # Move the exported .onnx into the crate so `cargo build` bundles it.
    dst = Path(__file__).parent.parent / "crates" / "vision" / "models" / "popup-yolov8n-v1.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, dst)
    print(f"\nModel copied to: {dst}")
    print("Update bw_vision::MODEL_PATH and rebuild bw-vision to use it.")


if __name__ == "__main__":
    main()
