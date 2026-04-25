"""
Score all unlabeled tasks from a Label Studio JSON export with the trained model.

Outputs predictions.csv sorted by uncertainty (most uncertain first) so you
know which images to label next to improve the model fastest.

How to get the export file:
  Label Studio → your project → Export → JSON → save as e.g. export.json
  (export ALL tasks, not just labeled ones)

Usage:
  python -m bisexual_lighting.predict export.json
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import torch
from botocore.client import Config
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .model import LightingClassifier

load_dotenv(override=True)

log = logging.getLogger(__name__)

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_BUCKET = os.environ["MINIO_BUCKET"]
MC_PASSES = int(os.environ.get("MC_PASSES", "20"))

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lighting_classifier.pt"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "predictions.csv"

# ImageFolder sorts classes alphabetically: no=0, yes=1
LABELS = ["no", "yes"]

INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def extract_s3_key(image_url: str) -> str:
    if image_url.startswith("s3://"):
        return urlparse(image_url).path.lstrip("/")
    path = urlparse(image_url).path.lstrip("/")
    if path.startswith(f"{MINIO_BUCKET}/"):
        path = path[len(MINIO_BUCKET) + 1:]
    return path


def load_image(s3, key: str) -> Image.Image:
    buf = io.BytesIO()
    s3.download_fileobj(MINIO_BUCKET, key, buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        log.error("Usage: python -m bisexual_lighting.predict <export.json>")
        log.error("Export JSON from Label Studio: Project → Export → JSON")
        sys.exit(1)

    export_path = Path(sys.argv[1])
    if not export_path.exists():
        log.error("File not found: %s", export_path)
        sys.exit(1)

    if not MODEL_PATH.exists():
        log.error("No model found at %s — run train.py first.", MODEL_PATH)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = LightingClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_tasks = json.loads(export_path.read_text(encoding="utf-8"))
    unlabeled = [t for t in all_tasks if not t.get("annotations")]
    log.info("Loaded %d tasks, %d unlabeled", len(all_tasks), len(unlabeled))

    s3 = get_s3_client()
    rows: list[dict] = []
    failed = 0

    for task in tqdm(unlabeled, desc="Scoring"):
        image_url = task["data"].get("image", "")
        if not image_url:
            failed += 1
            continue
        try:
            s3_key = extract_s3_key(image_url)
            image = load_image(s3, s3_key)
            tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(device)
            mean_probs, uncertainty = model.predict_with_uncertainty(tensor, n_passes=MC_PASSES)
            pred_idx = mean_probs.argmax(1).item()
            confidence = mean_probs[0, pred_idx].item()
            rows.append({
                "task_id": task["id"],
                "image": Path(s3_key).name,
                "predicted_label": LABELS[pred_idx],
                "confidence": round(confidence, 4),
                "uncertainty": round(uncertainty[0].item(), 6),
            })
        except Exception as exc:
            log.warning("Task %s failed: %s", task.get("id"), exc)
            failed += 1

    # Sort by uncertainty descending — label these first
    rows.sort(key=lambda r: r["uncertainty"], reverse=True)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task_id", "image", "predicted_label", "confidence", "uncertainty"]
        )
        writer.writeheader()
        writer.writerows(rows)

    log.info("Scored %d tasks (%d failed). Written to %s", len(rows), failed, OUTPUT_PATH)
    log.info(
        "Predicted positives: %d / %d",
        sum(1 for r in rows if r["predicted_label"] == "yes"),
        len(rows),
    )
    log.info("Label the tasks at the TOP of predictions.csv first (highest uncertainty).")


if __name__ == "__main__":
    main()
