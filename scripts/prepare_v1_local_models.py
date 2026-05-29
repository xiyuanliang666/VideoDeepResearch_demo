"""Prepare local OCR/object/CLIP model dependencies for VDR v1.

Run this from a normal terminal with network access, using the vdr-py311
environment. It intentionally initializes each optional local model once so
later benchmark runs do not fail mid-pipeline because a model download is
needed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local model assets for VDR v1.")
    parser.add_argument("--models-dir", default="models", help="Directory for local model files.")
    parser.add_argument("--yolo-model", default="yolo11n.pt", help="YOLO model filename or path.")
    parser.add_argument("--skip-paddleocr", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-open-clip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_yolo:
        from ultralytics import YOLO

        model_path = models_dir / args.yolo_model
        model = YOLO(str(model_path) if model_path.exists() else args.yolo_model)
        resolved = Path(str(getattr(model, "ckpt_path", "") or args.yolo_model))
        if resolved.exists() and not model_path.exists():
            shutil.copy2(resolved, model_path)
            resolved = model_path
        print(f"YOLO ready: {resolved}")

    if not args.skip_paddleocr:
        from paddleocr import PaddleOCR

        try:
            PaddleOCR(use_textline_orientation=True, lang="en")
        except Exception:
            PaddleOCR(lang="en")
        print("PaddleOCR ready")

    if not args.skip_open_clip:
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms("MobileCLIP-S2", pretrained="datacompdr")
        _ = model.eval()
        print("open_clip MobileCLIP-S2 ready")


if __name__ == "__main__":
    main()
