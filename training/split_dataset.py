"""
Move a fraction of train/ images + labels into val/ for YOLO eval.

    cd training/
    python split_dataset.py 0.1     # 10% → val

Idempotent: each image is moved only if the target file doesn't yet
exist. Re-running with the same fraction does nothing.
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
IMG_TRAIN = ROOT / "dataset" / "images" / "train"
LBL_TRAIN = ROOT / "dataset" / "labels" / "train"
IMG_VAL = ROOT / "dataset" / "images" / "val"
LBL_VAL = ROOT / "dataset" / "labels" / "val"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "fraction", type=float, nargs="?", default=0.1,
        help="fraction of training set to move to val (default 0.1 = 10 percent)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    IMG_VAL.mkdir(parents=True, exist_ok=True)
    LBL_VAL.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in IMG_TRAIN.glob("*.png"))
    if not images:
        print(f"No images in {IMG_TRAIN}")
        return
    random.seed(args.seed)
    random.shuffle(images)

    n_val = int(len(images) * args.fraction)
    moved = 0
    for img in images[:n_val]:
        lbl = LBL_TRAIN / (img.stem + ".txt")
        new_img = IMG_VAL / img.name
        new_lbl = LBL_VAL / lbl.name
        if new_img.exists() or not lbl.exists():
            continue
        shutil.move(str(img), str(new_img))
        shutil.move(str(lbl), str(new_lbl))
        moved += 1

    print(f"Moved {moved} / {len(images)} images to val/ ({args.fraction*100:.0f} percent)")


if __name__ == "__main__":
    main()
