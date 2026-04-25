"""
Download labeled images from a Label Studio JSON export into local dataset dirs.

Creates:
  data/bisexual_lighting/yes/  — images labeled as bisexual lighting
  data/bisexual_lighting/no/   — everything else

How to get the export file:
  Label Studio → your project → Export → JSON → save as e.g. export.json

Usage:
  python -m bisexual_lighting.export_labels export.json
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.client import Config
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

log = logging.getLogger(__name__)

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
MINIO_BUCKET = os.environ["MINIO_BUCKET"]
POSITIVE_LABEL = os.environ.get("POSITIVE_LABEL", "Yes")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bisexual_lighting"


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
    """Extract the S3 object key from whatever URL format Label Studio uses."""
    if image_url.startswith("s3://"):
        return urlparse(image_url).path.lstrip("/")
    path = urlparse(image_url).path.lstrip("/")
    if path.startswith(f"{MINIO_BUCKET}/"):
        path = path[len(MINIO_BUCKET) + 1:]
    return path


def parse_label(task: dict) -> str | None:
    """Return the first non-cancelled annotation choice, or None."""
    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue
        for result in annotation.get("result", []):
            choices = result.get("value", {}).get("choices", [])
            if choices:
                return choices[0]
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        log.error("Usage: python -m bisexual_lighting.export_labels <export.json>")
        log.error("Export JSON from Label Studio: Project → Export → JSON")
        sys.exit(1)

    export_path = Path(sys.argv[1])
    if not export_path.exists():
        log.error("File not found: %s", export_path)
        sys.exit(1)

    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    log.info("Loaded %d tasks from %s", len(tasks), export_path)

    labeled = [t for t in tasks if t.get("annotations")]
    log.info("%d tasks have annotations", len(labeled))

    s3 = get_s3_client()
    counts = {"yes": 0, "no": 0, "skipped": 0}

    for task in tqdm(labeled, desc="Downloading"):
        label = parse_label(task)
        if label is None:
            counts["skipped"] += 1
            continue

        subdir = "yes" if label == POSITIVE_LABEL else "no"
        out_dir = DATA_DIR / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        image_url = task["data"].get("image", "")
        if not image_url:
            counts["skipped"] += 1
            continue

        s3_key = extract_s3_key(image_url)
        out_path = out_dir / Path(s3_key).name

        if out_path.exists():
            counts[subdir] += 1
            continue

        try:
            buf = io.BytesIO()
            s3.download_fileobj(MINIO_BUCKET, s3_key, buf)
            out_path.write_bytes(buf.getvalue())
            counts[subdir] += 1
        except Exception as exc:
            log.warning("Failed to download %s: %s", s3_key, exc)
            counts["skipped"] += 1

    log.info(
        "Done.  yes=%d  no=%d  skipped=%d",
        counts["yes"], counts["no"], counts["skipped"],
    )


if __name__ == "__main__":
    main()
