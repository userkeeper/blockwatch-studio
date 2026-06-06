# Vision training workflow

This directory contains everything needed to train the YOLOv8n
detection model used by `bw-vision` at runtime. The pipeline lives
outside Cargo because (a) Ultralytics is Python-only and (b) we don't
want every Rust contributor to install PyTorch.

## Workflow

1. **Collect screenshots** of each class on a real Windows 11 box:

       cd ..
       cargo run --release -p bw-studio-cli -- --collect \
           wallet_receive_popup --out-dir dataset/images/train/

   Press `C` to save the current frame, `Esc` to exit. Aim for ~50
   examples per class, across light/dark themes, different DPI
   scales, and different window sizes.

2. **Label** them with [LabelImg](https://github.com/HumanSignal/labelImg)
   in YOLO format. The class IDs must match `training/dataset.yaml`
   (0 = wallet_receive_popup, 1 = seed_phrase_reveal, etc.). LabelImg
   writes one `.txt` per image into `dataset/labels/train/`.

3. **Split** train/val 90/10:

       python split_dataset.py 0.1   # not yet implemented

4. **Train**:

       python -m pip install ultralytics
       python train_yolov8.py

   ~30 min on a 8-core CPU for 80 epochs on 500 images.

5. **Export to ONNX**: handled automatically by `train_yolov8.py`.
   The script copies `best.onnx` into
   `crates/vision/models/popup-yolov8n-v1.onnx`.

6. **Wire the model into `bw_vision`**: open
   `crates/vision/src/lib.rs`, set `MODEL_PATH = "models/popup-
   yolov8n-v1.onnx"`, replace the `Error::ModelNotBundled` stub
   with a real `ort::Session::builder().commit_from_file(...)`.

## Files

| File              | Purpose                                  |
| ----------------- | ---------------------------------------- |
| `dataset.yaml`    | YOLO dataset config (paths + class names) |
| `train_yolov8.py` | Trains YOLOv8n, exports ONNX             |
| `README.md`       | This file                                |

## Class taxonomy

Mirrored from `bw_vision::ObjectClass`. Adding a class:

1. Add the variant to `ObjectClass` in Rust.
2. Add the name to `dataset.yaml` in the same order.
3. Collect ~50 labelled examples.
4. Retrain.
5. Bump model version (e.g. `popup-yolov8n-v2.onnx`) and update
   `MODEL_PATH`.
